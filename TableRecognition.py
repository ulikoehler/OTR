#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import cv_algorithms
import networkx as nx
import operator
import scipy.spatial.distance
from UliEngineering.SignalProcessing.Selection import multiselect
from UliEngineering.Utils.NumPy import invert_bijection

def transitive_closure(clusters):
    """
    Takes a map of <node number> => <list of associated nodes>
    and computes all transitive closures using a fast set-based algorithm

    Returns a list of sets (each set containing one transitive closure) and
    a node -> closure ID lookup table. The closure id can be used to look up
    the hull nodes as index to the list of closure
    """
    transitive_clusters = []
    cluster_lut = np.zeros(len(clusters), np.int) # Node ID -> cluster map
    already_assigned = np.zeros(len(clusters), np.bool)
    for i, cluster in clusters.items():
        if already_assigned[i]:
            continue
        # Build transitive list
        already_seen = {i} # Avoids infinite recursion, also this is the newly built transitive group
        todo = set(cluster)
        # Find all transitive elements in the current cluster
        while len(todo) > 0:
            j = todo.pop()
            jsim = clusters[j]
            # Try to avoid too much Python algorithms, use C core functions instead
            already_seen.add(j)
            already_seen |= set(jsim)
            todo |= set(jsim)
            todo -= already_seen
            # Set already seen array
            already_assigned[j] = True
            already_assigned[list(jsim)] = True
        # Now we have a list of all nodes in the sim group.
        # We can assign IDs (i.e. the index) and
        #  store it in a <endpoint ID> -> <cluster> LUT
        cluster_no = len(transitive_clusters) # Because we append once per loop
        transitive_clusters.append(already_seen)
        cluster_lut[list(already_seen)] = cluster_no
    return transitive_clusters, cluster_lut



def is_inside_table(polygon, reference):
    """
    Determine if a given polygon is fully located inside
    This works by checking if all polygon points are within (or on the edge of)
    the reference 

    returns True if and only if polygon is fully within or identical to the reference.
    """
    brect = cv2.boxPoints(cv2.minAreaRect(polygon))
    # All points need to be inside table corners
    for point in brect:
        if cv2.pointPolygonTest(reference, tuple(point), False) < 0:
            return False  # Not in contour
    return True

def angle_degrees(dx, dy):
    if dx == 0: return 180.
    return np.rad2deg(np.arctan2(dx, dy))


def _find_contours(*args, **kwargs):
    """
    Calls cv2.findContours(*args, **kwargs) and returns (contours, hierarchy)
    """
    tupl = cv2.findContours(*args, **kwargs)
    # Fix for #8
    if len(tupl) == 3:
        im2, contours, hierarchy = tupl
    elif len(tupl) == 2:
        contours, hierarchy = tupl
    return contours, hierarchy

class ContourAnalyzer(object):
    def __init__(self, img, **kwargs):
        contours, hierarchy = _find_contours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS, **kwargs)
        self.hierarchy = hierarchy
        self.contours = contours
        self.imgshape = img.shape

    @property
    def size(self):
        return len(self.contours)

    def build_graph(self):
        # Contour to graph
        self.g = nx.DiGraph()
        # Add each contour as node (by index)
        self.g.add_nodes_from(range(self.hierarchy.shape[1]))
        # Build graph from hierarchy
        for i in range(self.hierarchy.shape[1]):
            # Skip already invalidated contours
            if self.contours[i] is None: continue
            nxt, prev, first_child, parent = self.hierarchy[0, i]
            if parent != -1:
                self.g.add_edge(parent, i)

    def compute_cell_polygons(self):
        """
        Compute a list of cell polygons from the contours and the associated
        corner clusters. Generates a list of cells, with a cell being a list of
        cluster IDs. This list is stored in self.cell_polygons
        """
        cell_polygons = []
        for i in range(len(self.contours)):
            # Skip invalidated contours
            if self.contours[i] is None: continue
            bbox = self.contours_bbox[i]  # Guaranteed to have 4 nodes by design (it's a bounding box)!
            # bbox == corner1, ..., corner4
            bclust = multiselect(self.cluster_coords_to_node_id, bbox, convert=tuple) 
            cell_polygons.append(bclust)
        self.cell_polygons = cell_polygons


    def compute_cell_hulls(self):
        """
        Run find_table_cell_polygons() and compute a rectangle enclosing the cell (for each cell).
        For most (4-point) cells, this is equivalent to the original path, however this removes
        small irregularities and extra points from larger, 5+-point cells (mostly merged cells)
        """
        self.compute_cell_polygons()
        # cv2 convexHull / minAreaRect only work with integer coordinates.
        self.cell_hulls = [
            cv2.boxPoints(cv2.minAreaRect(np.rint(self.cluster_coords[path]).astype(int)))
            for path in self.cell_polygons]
        # Compute centers of cell hulls
        self.cell_centers = np.zeros((len(self.cell_hulls), 2))
        for i in range(len(self.cell_hulls)):
            hull_points = self.cell_hulls[i]
            self.cell_centers[i] = cv_algorithms.meanCenter(hull_points)

    def compute_table_coordinates(self, xthresh=8., ythresh=8.):
        """
        Sorts all clusters into spreadsheet-like x/y coordinates.
        Set self.cell_table_coord of shape (0, n)
        """
        center_x = self.cell_centers[:, 0]
        center_y = self.cell_centers[:, 1]
        # Compute adjacency list
        hgroup_adjlist, vgroup_adjlist = {}, {}
        for i in range(center_x.shape[0]):
            hgroup_adjlist[i] = np.nonzero(np.abs(center_x - center_x[i]) < xthresh)[0]
            vgroup_adjlist[i] = np.nonzero(np.abs(center_y - center_y[i]) < ythresh)[0]

        # Compute transitive closures so we get ALL grouped cells for each group,
        # not just the ones that are similar to the first node.
        hgroups, hgroup_lut = transitive_closure(hgroup_adjlist)
        vgroups, vgroup_lut = transitive_closure(vgroup_adjlist)
        ####
        # Reorder groups by x/y
        ####
        # Compute mean X/Y coords for each hgroup/vgroup
        hgroup_mean_centers = np.zeros(len(hgroups))
        vgroup_mean_centers = np.zeros(len(vgroups))
        for i, st in enumerate(hgroups):
            hgroup_mean_centers[i] = np.mean(self.cell_centers[list(st)][:, 0])
        for i, st in enumerate(vgroups):
            vgroup_mean_centers[i] = np.mean(self.cell_centers[list(st)][:, 1])
        # Find the sorted ordering (like in a spreadsheet) for both x and y coords
        hgroup_sort_order = np.argsort(hgroup_mean_centers)
        vgroup_sort_order = np.argsort(vgroup_mean_centers)
        # Output of argsort: Input => new index ; output => old index
        # BUT WE NEED: Input oldplace, output newplace
        hgroup_sort_order = invert_bijection(hgroup_sort_order)
        vgroup_sort_order = invert_bijection(vgroup_sort_order)
        # Reorder everything based on the new order
        hgroups = multiselect(hgroups, hgroup_sort_order)
        vgroups = multiselect(vgroups, vgroup_sort_order)
        # Convert LUTs
        hgroup_lut = hgroup_sort_order[hgroup_lut]
        vgroup_lut = vgroup_sort_order[vgroup_lut]

        # Build a (n, (tx,ty)) table coordinate LUT for all nodes
        self.cell_table_coord = np.dstack([hgroup_lut, vgroup_lut])[0]

        # Build index of table coordinates to node ID
        self.table_coords_to_node = {}
        for i, (x, y) in enumerate(self.cell_table_coord):
            self.table_coords_to_node[(x, y)] = i


    def filter_contours(self, min_area=250, min_nodes=4):
        """
        Remove contours:
            - With less than a specific area (square px)
            - With less than n nodes (usually 4)
        """
        for i in range(self.size):
            # Skip already invalidated contours
            if self.contours[i] is None: continue
            # Compute key parameters
            num_nodes = self.contours[i].shape[0]
            area = cv2.contourArea(self.contours[i])
            # Check if node shall be removed
            if area < min_area or num_nodes < min_nodes:
                self.contours[i] = None

    def remove_non_table_nodes(self):
        """
        Identify the topmost table node ("supernode") (i.e. the node with the most direct children)
        and delete every node in the graph (and invalidate any related contour)
        that is not either:
            - The supernode itself (i.e. the table outline) or
            - A direct child of the supernode
        This will remove stuff nested inside table cells and nodes outside the table
        The nodes are not removed from the graph.
        """
        self.supernode_idx = max(dict(self.g.degree()).items(), key=operator.itemgetter(1))[0]
        for i in range(len(self.contours)):
            if self.contours[i] is None: continue
            nxt, prev, first_child, parent = self.hierarchy[0, i]
            # Remove node if it has a non-supernode node as parent
            if parent != self.supernode_idx and i != self.supernode_idx:
                self.contours[i] = None
    
    def compute_contour_bounding_boxes(self):
        """
        Compute rotated min-area bounding boxes for every contour
        """
        self.contours_bbox = [None] * len(self.contours)
        self.aspect_ratios = np.zeros(self.size) # of rotated bounding boxes
        for i in range(len(self.contours)):
            if self.contours[i] is None: continue
            # Compute rotated bounding rectangle (4 points)
            bbox = cv2.minAreaRect(self.contours[i])
            # Compute aspect ratio
            (x1, y1), (x2, y2), angle = bbox
            self.aspect_ratios[i] = np.abs(x2 - x1) / np.abs(y2 - y1)
            # Convert to 4-point rotated box, convert to int and set as new contour
            self.contours_bbox[i] = np.rint(cv2.boxPoints(bbox)).astype(np.int)

    def separate_supernode(self):
        """
        Remove the supernode from the contours and save it separately.
        This means that only table cells and artifacts should be left as contours
        """
        # Store separately
        self.supernode = self.contours[self.supernode_idx]
        self.supernode_bbox = self.contours_bbox[self.supernode_idx]
        # Invalidate in normal storage
        self.contours[self.supernode_idx] = None
        self.contours_bbox[self.supernode_idx] = None

    def does_page_have_valid_table(self, min_fract_area=.2, min_cells=50):
        """
        Analyzes whether the image contains a table by evaluating the
        coarse table outline and its children
        """
        try: # Some CV2 operations may fail e.g. if no correct supernode has been recognized
            # Check fractional area of table compared to image
            img_area = self.imgshape[0] * self.imgshape[1]
            supernode_area = cv2.contourArea(self.supernode_bbox)
            if supernode_area < img_area * min_fract_area:
                return False
            # Check minimum number of cells (ncells = degree of coarse outline node)
            ncells = self.g.degree(self.supernode_idx)
            return ncells >= min_cells
        except cv2.error:
            return False

    def find_empty_cells(self, img, threshold=.998):
        """
        Find out which cells are empty by 
        """
        # Compute which cells are empty
        self.is_contour_empty = np.zeros(self.size, np.bool) # of rotated bounding boxes
        for i in range(len(self.contours)):
            if self.contours[i] is None: continue
            sect = cv_algorithms.extractPolygonMask(img, self.contours_bbox[i])
            self.is_contour_empty[i] = cv_algorithms.fractionWhite(sect) > threshold
    
    def compute_contour_centers(self):
        """
        Compute cell centers for each contour bounding box using meanCenter()
        """
        self.contour_centers = np.full((len(self.contours), 2), -1.)  # -1: Contour invalid
        for i in range(len(self.contours)):
            if self.contours[i] is None: continue
            self.contour_centers[i] = cv_algorithms.meanCenter(self.contours_bbox[i])

    def find_corner_clusters(self, distance_threshold=20.):
        # Find all bounding box corners
        corners = []
        for i in range(len(self.contours)):
            if self.contours[i] is None: continue
            bbox = self.contours_bbox[i]
            for coord in bbox:
                corners.append((coord[0], coord[1]))

        # Simpler algorithm, still superfast (<40 ms for 2k corners): Compute all distances using cdist
        corners = np.asarray(corners)
        distmat = scipy.spatial.distance.cdist(corners, corners, 'euclidean')
        ignore = np.zeros(corners.shape[0], np.bool) # Set to true if we found a cluster for this node already

        # Find cluster in the distance matrix, i.e. node groups which are close together
        cluster_coords = []  # For each cluster, a (x,y coordinate pair)
        cluster_num_nodes = []  # For each cluster, the number of nodes it consists of
        cluster_coords_to_node_id = {}  # (x,y) tuple => cluster ID
        for i in range(corners.shape[0]):
            if ignore[i]: continue
            # Which nodes are close to this node, including itself
            below_thresh = distmat[i, :] < distance_threshold # Rather set this large, we can correct non-convexity later
            allnodes = np.nonzero(below_thresh)[0] # index list
            # Get a new ID
            clusterid = len(cluster_coords)
            allcorners = corners[allnodes]
            cluster_coords.append(tuple(cv_algorithms.meanCenter(allcorners)))
            cluster_num_nodes.append(allnodes.size)
            # Also create a map from each position to the current cluster ID
            # This works only because these coordinates are discrete integer pixel indices
            for coord in allcorners:
                cluster_coords_to_node_id[tuple(coord)] = clusterid
            # Ignore all nodes in the cluster (i.e. don't assign them to a new cluster)
            ignore[allnodes] = True
        # Now that the size is known, we can convert to numpy arrays
        self.cluster_coords = np.asarray(cluster_coords)
        self.cluster_num_nodes = np.asarray(cluster_num_nodes)
        self.cluster_coords_to_node_id = cluster_coords_to_node_id

    def find_fine_table_corners(self):
        """
        Find fine table corners. This function works on cluster coordinates.
        It returns the ones with the most extreme coordinates, intermixing X/Y coordinates.
        """
        # The absolute min/max (top left / bot right corner) can be found without array modification3
        minmax_unmodified = np.prod(self.cluster_coords, axis=1)
        minx_miny = np.argmin(minmax_unmodified)
        maxx_maxy = np.argmax(minmax_unmodified)
        # Copy and modify the array
        ccopy = self.cluster_coords.copy()
        ccopy[:, 1] = np.reciprocal(ccopy[:, 1])  # multiply Y by -1, results in X/-Y
        minx_maxy = np.argmin(np.prod(ccopy, axis=1))

        ccopy = self.cluster_coords.copy()
        ccopy[:, 0] = np.reciprocal(ccopy[:, 0])  # multiply X/Y by -1, results in -X/Y
        maxx_miny = np.argmin(np.prod(ccopy, axis=1))

        corners = np.rint(self.cluster_coords[[minx_miny, maxx_miny, maxx_maxy, minx_maxy, minx_miny]]).astype(np.int)
        self.table_corners = corners
        return corners

    def compute_missing_cells_mask(self, close_ksize=5):
        """
        Compute a binary img-scale mask,
        """
        # Create white binary img
        icellmask = np.full((self.imgshape[0], self.imgshape[1]), 255, np.uint8)

        # Mask everything except table, as defined by corner nodes (not the larger super-node!)
        cv2.fillConvexPoly(icellmask, self.table_corners, 0)
        # Now draw all cell hulls without text, but don't downsize them()
        self.draw_all_cell_hulls(icellmask, None, xscale=1.1, yscale=1.1)

        # Morphology ops with large kernel to remove small intercell speckles
        # NOTE: CLOSE => remove black holes
        icellmask = cv2.morphologyEx(icellmask, cv2.MORPH_CLOSE,
                                     np.ones((close_ksize, close_ksize), np.uint8))
        return icellmask

    def compute_missing_cell_contours(self, missing_cells_mask):
        contx, _ = _find_contours(missing_cells_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        return contx

    def compute_filtered_missing_cell_contours(self):
        """
        Filter the given missing
        """
        missing_cells_mask = self.compute_missing_cells_mask()
        missing_cell_contours = self.compute_missing_cell_contours(missing_cells_mask)
        table_corners = self.table_corners  # fine table corners
        # Perform filtering
        return [c for c in missing_cell_contours if is_inside_table(c, table_corners)]


    def extract_cell_from_image(self, img, table_coords, xscale=3, yscale=3, mark_color=(255,0,0), mark_thickness=1):
        """
        Extract a section from the image that corresponds to the table cell.

        Parameters
        ----------
        img : array_like 2D
            The return value is a subsection of this
        table_coords : (x,y) table coordinates
            The coordinates of the cell to extract
        """
        # Convert table coordinates to node ID
        node_id = self.table_coords_to_node[table_coords]
        # Get upright bounding rectangle
        bounding_rect = cv2.boundingRect(self.cell_hulls[node_id])
        # Expand rectangle
        bounding_rect_expanded = cv_algorithms.expandRectangle(bounding_rect, xscale, yscale)
        # Slice image
        x, y, w, h = bounding_rect_expanded
        img_sect = img[y:y + h, x:x + w]
        # Mark original rectangle
        # The coordinates need to be re-calculated
        #  because img_sect has a different origin point than img.
        if mark_color is not None:
            x1, y1, w1, h1 = bounding_rect
            x2, y2, w2, h2 = bounding_rect_expanded
            cv2.rectangle(img_sect, (x1 - x2, y1 - y2),
                          (x1 - x2 + w1, y1 - y2 + h1), color=(0, 0, 255),
                          thickness=mark_thickness)
        return img_sect



    def visualize_corner_clusters(self, img):
        col_map = {0: (0, 0, 0),
                   1: (0, 0, 200),
                   2: (200, 0, 0),
                   3: (200, 200, 0),
                   4: (0, 200, 0),
                   5: (200, 200, 200),
                   -1: (255, 255, 255)}
        for i, (x, y) in enumerate(self.cluster_coords):
            n = self.cluster_num_nodes[i]
            col = col_map[n] if n < 5 else col_map[-1]
            cv2.circle(img, (int(x), int(y)), 8,
                       color=col, thickness=-1) # filled


    def visualize_contours(self, img, thickness=2, draw_empty=True):
        """
        Draw the contour-based table information
            - Outline in red
            - Cells in green
        """
        # Draw table outline in red
        cv2.drawContours(img, [self.supernode_bbox], -1, (255, 0, 0), thickness)
        # Draw table cells
        cv2.drawContours(img, self.contours_bbox, -1, (0,255,0), thickness)
        # Draw empty table cells (they have already been drawn as table cells, just draw over that)
        if draw_empty:
            cv2.drawContours(img, [c for i,c in enumerate(self.contours_bbox)
                                   if self.is_contour_empty[i] and c is not None], -1, (0, 0, 255), thickness)

    def draw_cell_hull(self, img, i, text, xscale=1., yscale=1., textsize=1, fill_color=(255,255,0)):
        """
        Draw a single cell hull as a filled polygon
        """
        # Drawing functions need pure integer coordinates
        thehull = np.rint(cv_algorithms.scaleByRefpoint(self.cell_hulls[i], xscale, yscale)).astype(np.int)
        # Draw the polygon
        cv2.fillConvexPoly(img, thehull, color=fill_color)
        # Draw text
        if text is not None:
            thehull_center = np.mean(thehull, axis=0)
            boxWidth, boxHeight = np.ptp(thehull, axis=0) # Max X/Y extent of the box
            cv_algorithms.putTextAutoscale(img, text, thehull_center, cv2.FONT_HERSHEY_SIMPLEX,
                                           boxWidth, boxHeight, color=(0,0,0), thickness=2)
            
    def draw_all_cell_hulls(self, img, text="{0}", **kwargs):
        for i in range(len(self.cell_hulls)):
            self.draw_cell_hull(img, i, text.format(i) if text is not None else None, **kwargs)

    def draw_table_coord_cell_hulls(self, img, **kwargs):
        for i in range(len(self.cell_hulls)):
            tcoordX, tcoordY = self.cell_table_coord[i]
            text = "{0},{1}".format(tcoordX, tcoordY)
            self.draw_cell_hull(img, i, text, **kwargs)

    def visualize_node_arrows(self, img, node_id, size=3, color=(255,0,255)):
        l, r, t, b = self.adjacency[node_id]
        srcX, srcY = self.node_coordinates[node_id]
        for direction, r in enumerate((l,r,t,b)):
            if r == -1: continue
            targetX, targetY = self.node_coordinates[r]
            # Use constant arrow head size. As arrowedLine() takes a fraction of the length, we need to reverse that
            length = np.hypot(targetX - srcX, targetY - srcY)
            arrowHeadSizeWant = 15  #px
            arrowHeadSize = arrowHeadSizeWant / length
            print("Drawing <{3}> arrow from #{0} to #{1}  of length {2}".format(
                    r, node_id, length, {0: "left", 1: "right", 2:"top", 3:"bottom"}[direction]))
            cv2.arrowedLine(img, (int(targetX), int(targetY)), (int(srcX), int(srcY)),
                            color=color, thickness=3, tipLength=arrowHeadSize, line_type=cv2.LINE_AA)


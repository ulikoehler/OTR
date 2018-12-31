# OTR
Optical table recognition - recognize tables in scan images using OpenCV

# Install

Install OpenCV for Python3 e.g. `sudo apt-get install python3-opencv` in Ubuntu 18.04.

Dependency: ![cv_algorithms](https://github.com/ulikoehler/cv_algorithms)
```sh
sudo pip3 install git+https://github.com/ulikoehler/cv_algorithms.git
```

# Run

Get a test image, e.g. google for images like "Old naval log table" and select one with a table. Can't share one here due to copyright but if you know a public domain one, please add it via a pull request.

```sh
python3 test-otr.py <image filename>
```

It's currently only a proof of concept. See ![Algorithm.pdf](https://github.com/ulikoehler/OTR/blob/master/doc/Algorithm.pdf) for details on how it works.

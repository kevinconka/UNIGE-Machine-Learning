Semeion Handwritten Digit data

1593 handwritten digits from around 80 persons were scanned, 
stretched in a rectangular box 16x16 in a binary scale.

The dataset was downloaded from the U.C.I. Machine Learning Repository
www.ics.uci.edu/~mlearn/

As is standard in that collection, data are described in a plain text
file titled "semeion.names" and stored in a plain text file (in a 1593x266
table) called "semeion.data".

Each pattern is a 16x16 matrix of binary (black/white) values.
For more details read the ".names" file.

Additions w.r.t. the original dataset:
- added PNG images for each pattern (pixel values inverted to have black
  strokes on white background); they are in directory png
- added PNG images for the target digits, so that your classifier can show
  the "ideal" answer; also in directory png
- added a Matlab input script called "readdigits" that:
  - reads the patterns into a matrix x
  - appends a column of ones to x
  - reads the targets into a matrix t
  - moves class 0 to the last position, so that index 1 is class "1",
    index 2 is class "2", ..., index 10 is class "0" ("Matlab-friendly").

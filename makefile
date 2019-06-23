CC = g++
CFLAGS = -g -Wall
SRC = main.cpp
OUT = FaceRecognition

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

$(OUT):$(SRC)
		$(CC) $(CFLAGS) -o $(OUT) $(SRC) $(LIBS)
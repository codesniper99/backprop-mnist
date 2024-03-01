CC = g++
CFLAGS = -O3 -std=c++17 -Wall
TARGET = fnn-revised
SRCS = fnn-revised.cpp
OBJS = $(SRCS:.cpp=.o)
ARGS = 2 10 20 10 3

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -f $(OBJS) $(TARGET)
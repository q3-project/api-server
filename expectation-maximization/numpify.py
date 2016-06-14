import numpy
from PIL import Image

def PIL2array(img):
    return numpy.array(img.getdata(),
                    numpy.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = numpy.c_[arr, 255*numpy.ones((len(arr),1), numpy.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def main():
    img = loadImage('foo.jpg')
    arr = PIL2array(img)
    img2 = array2PIL(arr, img.size)
    img2.save('out.jpg')

if __name__ == '__main__':
    main()

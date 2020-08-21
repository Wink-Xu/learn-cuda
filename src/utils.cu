#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

using namespace cv;

Mat Array2Mat(unsigned char *array, int row, int col)
{
 
    Mat img(row ,col,  CV_8UC1);
    for (int i = 0; i <row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            img.at<unsigned char>(i, j) =  array[i * col + j] ;
        }
    }
    return img;
}

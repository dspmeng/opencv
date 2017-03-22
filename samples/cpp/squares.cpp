// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

#define VIS(feature)           \
if (vis) {                     \
    imshow(wndname, feature);  \
    int key = waitKey();       \
    if (key == 27) {           \
        destroyAllWindows();   \
        exit(0);               \
    }                          \
}

static void help()
{
    cout <<
    "Usage: cpp-example-squares [--image=IMAGE_TO_DETECT] [--ratio=MIN/MAX_AREA_RATIO]\n"
    "\n"
    "A program using pyramid scaling, Canny, contours, contour simpification\n"
    "to find squares in a list of preset images and the user provided image\n"
    "Returns sequence of squares detected on these images.\n"
    "Using OpenCV version: " << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";
float minratio = 4.0;
float maxratio = 4.0;
bool vis = false;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    VIS(timg);

    vector<vector<Point> > contours;


    // find squares in every color plane of the image
    for (int c = 0; c < 3; c++)
    {
        // pick one of r/g/b channels
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
        VIS(gray0);

        // try several threshold levels
        for (int l = 0; l < N; l++)
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if (l == 0)
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 3);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
                VIS(gray);
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
                VIS(gray);
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for (size_t i = 0; i < contours.size(); i++)
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // excluding too small or too large areas (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if (approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > (image.cols * image.rows / 24 / minratio) &&
                    fabs(contourArea(Mat(approx))) < (image.cols * image.rows / 24 * maxratio) &&
                    isContourConvex(Mat(approx)))
                {
                    double maxCosine = 0;

                    for (int j = 2; j < 5; j++)
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares(Mat& image, const vector<vector<Point> >& squares)
{
    for (size_t i = 0; i < squares.size(); i++) {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        for (int j = 0; j < n; j++)
            squares[i][j];
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 1, LINE_AA);
    }

    imshow(wndname, image);
}


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
        "{help h    |     | print this message}"
        "{image     |     | image for detection}"
        "{minratio  | 4.0 | min color block area img_area/24/minratio}"
        "{maxratio  | 4.0 | max color block area img_area/24*maxratio}"
        "{vis       |     | visualize feature maps}");

    if (parser.has("help")) {
        help();
        parser.printMessage();
    }

    const char* array[] = {
        "../data/IQTest_Colorchecker_HDR_D50_40.jpg",
        "../data/IQTest_Colorchecker_HDR_D50_40-rot.jpg",
        "../data/IQTest_Colorchecker_HDR_D50_40-affine.jpg",
        "../data/blob.png",
        "../data/IntelInddor25fps0307_Jeff2.avi-012.png",
        "../data/contour.png",
        "../data/HDRScene_30fps_3.avi-011.png",
        "../data/hdr.png"};
    vector<string> names(array, array + sizeof(array) / sizeof(array[0]));
    if (parser.has("image")) {
        names.insert(names.begin(), parser.get<String>("image"));
    }

    if (parser.has("minratio")) {
        minratio = parser.get<float>("minratio");
    }

    if (parser.has("maxratio")) {
        maxratio = parser.get<float>("maxratio");
    }

    if (parser.has("vis")) {
        vis = true;
    }

    namedWindow(wndname);
    vector<vector<Point> > squares;

    for (size_t i = 0; i < names.size(); i++ )
    {
        Mat image = imread(names[i], 1);
        if (image.empty())
        {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        VIS(image);
        findSquares(image, squares);
        drawSquares(image, squares);

        int c = waitKey();
        if ((char)c == 27)
            break;
    }

    return 0;
}

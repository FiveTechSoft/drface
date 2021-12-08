/*
  OpenCV and Face recognition
  c)copyright 2021 - By Charles KWON

   FWH64 + Harbour

*/

#include "fivewin.ch"

CLASS TCV2

    DATA cCascadeName
    DATA cNestedCascadeName

    DATA nFaces
    DATA aFaces
    DATA aFeatures

    METHOD New() CONSTRUCTOR
    METHOD imread( cFile, nFlag )

    METHOD waitKey( nSecond )
    METHOD imShow( cTitle, pImage )
    METHOD DeleteObject( oObj ) INLINE __IMDESTROY( oObj )

    METHOD DetectFaces( pImg, nFaces, aFaces )
    METHOD MatchFaces( cFeature1, cFeature2, @nL2Score )

    METHOD DrawFaces( hDC, aFaces )
    METHOD DrawRect( hDC, nTop, nLeft, nHeight, nWidth, nColor, nDepth )

ENDCLASS


METHOD New() CLASS TCV2

RETURN Self

METHOD imread ( cFile, nFlag, pImage ) CLASS TCV2
    LOCAL lOk

    lOk := cvImread ( cFile, nFlag, @pImage )

RETURN lOk

 METHOD waitKey( nSecond ) CLASS TCV2

    CVWaitKey ( nSecond )

 RETURN NIL

 METHOD imShow( cTitle, pImage ) CLASS TCV2

    DEFAULT cTitle := ""
    CVIMSHOW( cTitle, pImage )

 RETURN NIL


METHOD DetectFaces( pImg, nFaces, aFaces, aFeatures ) CLASS TCV2
    LOCAL lOk

    lOk := DetectFaces( pImg, @nFaces, @aFaces, @aFeatures )

    ::nFaces := nFaces
    ::aFaces := aFaces

RETURN lOk

METHOD MatchFaces( cFeature1, cFeature2, nL2Score ) CLASS TCV2
    LOCAL nCos_score
                                                     
    nCos_score := MatchFaces( cFeature1, cFeature2, @nL2Score )
    
RETURN round( nCos_score, 8 )


 METHOD DrawFaces( hDC, aFaces ) CLASS TCV2
    LOCAL nI

    FOR nI := 1 TO LEN( aFaces )
        ::DrawRect( hDC, aFaces[nI][2], aFaces[nI][1], aFaces[nI][4], aFaces[nI][3], CLR_HGREEN, 2 )
    NEXT

 RETURN NIL


 METHOD DrawRect( hDC, nTop, nLeft, nHeight, nWidth, nColor, nDepth ) CLASS TCV2
   LOCAL hDarkPen
   LOCAL hOldObject
   LOCAL nBottom := nTop + nHeight
   LOCAL nRight  := nLeft + nWidth

   hOldObject := SelectObject( hDC, GetStockObject( 5 ) )

   hDarkPen  = CreatePen( PS_SOLID, nDepth, nColor )
   Rectangle( hDC, nTop, nLeft, nBottom, nRight, hDarkPen )

   DeleteObject( hDarkPen )
   SelectObject( hDC, hOldObject )

RETURN NIL

INIT FUNCTION __DrFaceInit()

    __DrfaceInit__()

RETURN NIL


#pragma BEGINDUMP

#include <opencv2/objdetect.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>



#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>

#include <Windows.h>

#include <hbapi.h>
#include "hbapiitm.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;



void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );


string cascadeName;
string nestedCascadeName;


static Ptr<FaceDetectorYN> detector;
static Ptr<FaceRecognizerSF> faceRecognizer;

/*
  Preload for speed
*/
HB_FUNC( __DRFACEINIT__ )
{
  String fd_modelPath = "yunet.onnx";
  String fr_modelPath = "face_recognizer_fast.onnx";
  
  double scoreThreshold;
  double nmsThreshold;

  int topK = 5000;
  bool save = false ;
  double cosine_similar_thresh = 0.363;
  double l2norm_similar_thresh = 1.128;

  scoreThreshold = 0.9 ;
  nmsThreshold = 0.3;
  

  detector = FaceDetectorYN::create(fd_modelPath, "", Size(320, 320), (float)scoreThreshold, (float)nmsThreshold, topK);
  faceRecognizer = FaceRecognizerSF::create(fr_modelPath, "");

}





HB_FUNC( TEST_DARK )
{

string cFile = (string)hb_parc(1);

cv::Mat img = cv::imread(cFile);


const float confidenceThreshold = 0.5f;
Net m_net;
std::string yolo_cfg = "yolov4.cfg";
std::string yolo_weights = "yolov4.weights";
m_net = readNetFromDarknet(yolo_cfg, yolo_weights);
m_net.setPreferableBackend(DNN_BACKEND_OPENCV);
m_net.setPreferableTarget(DNN_TARGET_CPU);

cv::Mat inputBlob = blobFromImage(img, 1 / 255.F, cv::Size(416, 416), cv::Scalar(), true, false); //Convert Mat to batch of images

m_net.setInput(inputBlob);

std::vector<cv::Mat> outs;

cv::Mat detectionMat = m_net.forward();

for (int i = 0; i < detectionMat.rows; i++)
{
  const int probability_index = 5;
  const int probability_size = detectionMat.cols - probability_index;
  float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
  size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
  float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
  if (confidence > confidenceThreshold)
  {
    float x_center = detectionMat.at<float>(i, 0) * (float)img.cols;
    float y_center = detectionMat.at<float>(i, 1) * (float)img.rows;
    float width = detectionMat.at<float>(i, 2) * (float)img.cols;
    float height = detectionMat.at<float>(i, 3) * (float)img.rows;
    cv::Point2i p1(round(x_center - width / 2.f), round(y_center - height / 2.f));
    cv::Point2i p2(round(x_center + width / 2.f), round(y_center + height / 2.f));
    cv::Rect2i object(p1, p2);

    cv::rectangle(img, object, cv::Scalar(0, 0, 255), 2);
  }
}

cv::imshow("img", img);
cv::waitKey();

return;
}







/*
  pImg := _cvImread("test.jpg",IMREAD_COLOR, @pStr )
*/
HB_FUNC( CVIMREAD )
{
  LPSTR lpName = (LPSTR) hb_parc(1);
  int iFlag  = hb_parni(2);

  PHB_ITEM pItem = hb_param( 3, HB_IT_ANY );
  Mat *__img = new Mat(imread( lpName, iFlag ));

  hb_itemPutPtr( pItem, __img);

  hb_retl( !__img->empty() ) ;
}


HB_FUNC( _CVIMREAD )
{
  LPSTR lpName = (LPSTR) hb_parc(1);
  int iFlag  = hb_parni(2);
  int iPos   =  hb_parni(3);

  static Mat image[100] ;

  image[iPos] = imread( lpName, iFlag );

  hb_retptr(  &image[iPos] );
}


HB_FUNC( CVIMSHOW )
{
  LPSTR lpTitle = (LPSTR) hb_parc(1) ;
  Mat* img = static_cast<Mat *>( hb_parptr(2));

  Mat image = img->clone();

  imshow( lpTitle, image);
}

HB_FUNC( __IMDESTROY )
{
  Mat* img = static_cast<Mat *>( hb_parptr(1));

  if(img!=NULL)
    delete img;

}


HB_FUNC( CVWAITKEY )
{
   int iSecond = hb_parni(1);
   waitKey( iSecond ) ;
}

HB_FUNC( INIT_OPENCV )
{
 Mat* img = static_cast<Mat *>( hb_parptr(1) );
 Mat image = img->clone();

 CascadeClassifier cascade, nestedCascade;
 double scale;

 imshow("Result", image);

 waitKey(0);
return;




 CoInitialize(NULL);

 cascadeName = "q:/fwh64/samples/haarcascade_frontalface_alt.xml";
 nestedCascadeName = "q:/fwh64/samples/haarcascade_eye_tree_eyeglasses.xml";
 scale = 1.3;

 if (!nestedCascade.load(samples::findFileOrKeep(nestedCascadeName)))
 {
      MessageBox( 0, "Error1","Error", MB_OK );
      return ;
 }


 if (!cascade.load(samples::findFileOrKeep(cascadeName)))
 {
     MessageBox( 0, "Error2","Error", MB_OK );
     return ;
 }


 //image = imread("test.jpg", IMREAD_COLOR );

 detectAndDraw( image, cascade, nestedCascade, scale, 0 );
 //detectAndDraw( imgtest, cascade, nestedCascade, scale, 0 );

 waitKey(0);


 //imshow("Result", image);
 // waitKey(0);

}


HB_FUNC( LOAD_NESTEDCASCADE )
{
    LPSTR lpFile = (LPSTR) hb_parc(1);
    CascadeClassifier nestedCascade;

    if (!nestedCascade.load(samples::findFileOrKeep( lpFile )))
 {
       MessageBox( 0, "Error1","Error", MB_OK );
      return ;
 }

}

static
void visualize(Mat& input, int frame, Mat& faces, double fps, int thickness = 2)
{
    std::string fpsString = cv::format("FPS : %.2f", (float)fps);
    if (frame >= 0)
        cout << "Frame " << frame << ", ";
    cout << "FPS: " << fpsString << endl;
    for (int i = 0; i < faces.rows; i++)
    {
        // Print results
        cout << "Face " << i
             << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
             << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
             << "score: " << cv::format("%.2f", faces.at<float>(i, 14))
             << endl;

        // Draw bounding box
        rectangle(input, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(0, 255, 0), thickness);
        // Draw landmarks
        circle(input, Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, Scalar(255, 0, 0), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, Scalar(0, 0, 255), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, Scalar(0, 255, 0), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, Scalar(255, 0, 255), thickness);
        circle(input, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, Scalar(0, 255, 255), thickness);
    }
    putText(input, fpsString, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);





}

HB_FUNC( DETECTFACES )
{
  Mat* img = static_cast<Mat *>( hb_parptr(1) );
  PHB_ITEM nCount = hb_param(2,HB_IT_ANY);
  PHB_ITEM aFaces = hb_param(3,HB_IT_ARRAY);
  PHB_ITEM aFeatures = hb_param(4,HB_IT_ARRAY);

  PHB_ITEM nFPS  = hb_param(5,HB_IT_ANY );

  PHB_ITEM Subarray   = hb_itemNew(NULL);
  PHB_ITEM Subarray2  = hb_itemNew(NULL);
  PHB_ITEM cTemp      = hb_itemNew(NULL);
  uchar * ucData ;
  int nSize ;


  Mat image1 = img->clone();
  String fd_modelPath = "yunet.onnx";
  String fr_modelPath = "face_recognizer_fast.onnx";

  bool save = false ;
  double cosine_similar_thresh = 0.363;
  double l2norm_similar_thresh = 1.128;

  

  TickMeter tm;

  tm.start();

  //! [inference]
  // Set input size before inference
  detector->setInputSize(image1.size());

  Mat faces1;
  detector->detect(image1, faces1);

  if (faces1.rows < 1)
  {
    hb_itemPutNI( nCount, 0 );
    hb_retl(false);
    return ;
  }

  tm.stop();
  hb_itemPutND( nFPS, tm.getFPS());

  hb_itemPutNI( nCount, faces1.rows );

  for (int i = 0; i < faces1.rows; i++)
  {
      hb_arrayNew( Subarray, 15 );

      hb_arraySetNInt( Subarray,  1, int(faces1.at<float>(i,  0) ) );
      hb_arraySetNInt( Subarray,  2, int(faces1.at<float>(i,  1) ) );
      hb_arraySetNInt( Subarray,  3, int(faces1.at<float>(i,  2) ) );
      hb_arraySetNInt( Subarray,  4, int(faces1.at<float>(i,  3) ) );
      hb_arraySetNInt( Subarray,  5, int(faces1.at<float>(i,  4) ) );  // right eye, x
      hb_arraySetNInt( Subarray,  6, int(faces1.at<float>(i,  5) ) );  // right eye, y
      hb_arraySetNInt( Subarray,  7, int(faces1.at<float>(i,  6) ) );  // left eye, x
      hb_arraySetNInt( Subarray,  8, int(faces1.at<float>(i,  7) ) );  // left eye, y
      hb_arraySetNInt( Subarray,  9, int(faces1.at<float>(i,  8) ) );  // nose tip, x
      hb_arraySetNInt( Subarray, 10, int(faces1.at<float>(i,  9) ) );  // nose tip, y
      hb_arraySetNInt( Subarray, 11, int(faces1.at<float>(i, 10) ) );  // right corner of mouth, x
      hb_arraySetNInt( Subarray, 12, int(faces1.at<float>(i, 11) ) );  // right corner of mouth, y
      hb_arraySetNInt( Subarray, 13, int(faces1.at<float>(i, 12) ) );  // left corner of mouth, x
      hb_arraySetNInt( Subarray, 14, int(faces1.at<float>(i, 13) ) );  // left corner of mouth, y

      hb_arrayAddForward( aFaces, Subarray );
  }

  /*
    feature extract
  */
  Mat aligned_face1 ;
  Mat feature1;

  for (int i = 0; i < faces1.rows; i++)
  {
    faceRecognizer->alignCrop(image1, faces1.row(i), aligned_face1);
    faceRecognizer->feature(aligned_face1, feature1);

   // feature1 = feature1.clone();

    nSize = 512; // feature1.total() * feature1.elemSize();
    ucData = new uchar[ nSize ] ;
    
    memcpy( ucData, feature1.data, 512 ) ;//nSize * sizeof(uchar) ) ;

    hb_arrayNew( Subarray2, 5 );

    hb_arraySetNInt( Subarray2, 1, (int)feature1.rows) ;
    hb_arraySetNInt( Subarray2, 2, (int)feature1.cols) ;
    hb_arraySetNInt( Subarray2, 3, (int)feature1.type()) ;
    hb_arraySetNInt( Subarray2, 4, (int)nSize) ;
    hb_arraySetCL( Subarray2, 5, (const char * )ucData,512 );

    hb_arrayAddForward( aFeatures , Subarray2);
  }


  hb_retl(true);


 // visualize(image1, 1, faces1, tm.getFPS(),  2);

}


void MsgBox( double d)
{
    char buff[100];
    

    sprintf_s(buff, "value is:%f",d);

    MessageBox(0, (LPSTR)buff, "hello", MB_OK );
}

/*

  Undocumented

  OpenCV Face feature Mat format :
    rows :   1
    cols : 128
    type :   5
    size : 512

*/
HB_FUNC( MATCHFACES )
{
   PHB_ITEM cFace1    = hb_param( 1, HB_IT_STRING );
   PHB_ITEM cFace2    = hb_param( 2, HB_IT_STRING );
   PHB_ITEM nL2Score  = hb_param( 3, HB_IT_ANY );


   uchar * ucData1 = (uchar *) hb_itemGetCPtr(cFace1);
   uchar * ucData2 = (uchar *) hb_itemGetCPtr(cFace2);

   Mat cFeature1 ;//= Mat(1,128, 5 , (byte *)ucData2).clone();
   Mat cFeature2 ;//= Mat(1,128, 5 , (byte *)ucData2).clone();

   cFeature1.create(1,128, 5);
   memcpy( cFeature1.data, ucData1, 512 ) ;
  // cFeature1 = cFeature1.clone();

   cFeature2.create(1,128, 5);
   memcpy( cFeature2.data, ucData2, 512 ) ;
  // cFeature2 = cFeature2.clone();

   double cos_score = faceRecognizer->match(cFeature1, cFeature2, FaceRecognizerSF::DisType::FR_COSINE);
   double L2_score = faceRecognizer->match(cFeature1, cFeature2, FaceRecognizerSF::DisType::FR_NORM_L2);


   //MsgBox(cos_score);
   //MsgBox(L2_score);

   hb_itemPutND( nL2Score, (DOUBLE) L2_score );

   hb_retnd( cos_score );
  

}

HB_FUNC( TESTCODE )
{
  // faceTest();

}




void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    double t = 0;
    std::vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );

    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( std::vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        std::vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            cv::rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
    imshow( "result", img );
}

#pragma ENDDUMP
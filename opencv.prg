/*
  OpenCV and Face recognition
  c)copyright 2021 - By Charles KWON

   FWH64 + Harbour

*/

#include "fivewin.ch"



FUNCTION Main()
    LOCAL oDlg
    LOCAL oImage
    LOCAL cFile := "test4.jpg"
    LOCAL oFindBut
    LOCAL oFindLandmarkBut

    SET DECIMAL TO 12

    //TEST_DARK("charles_s.jpg")
    //TEST_TRACKING()
  
    DEFINE DIALOG oDlg RESOURCE "D_DRFACE"

           REDEFINE IMAGE oImage FILE cFile ID 501 OF oDlg

           REDEFINE BUTTON oFindBut         ID 210 OF oDlg ACTION doFaceDetect( oDlg, cFile, oImage )           
           REDEFINE BUTTON oFindLandmarkBut ID 220 OF oDlg ACTION doFaceDetect( oDlg, cFile, oImage, .T. )           

    ACTIVATE DIALOG oDlg CENTERED


 

    
/*
    DEFINE DIALOG oDlg TITLE "DrFace   c)Charles KWON"

           @ 0,0 IMAGE oImage FILE cFile OF oDlg

           oDlg:bStart := {|| doFaceDetect( oDlg, cFile, oImage )  }

    ACTIVATE DIALOG oDlg CENTER
*/
RETURN NIL


FUNCTION doFaceDetect( oDlg, cFile, oImage, lLandmark )
    LOCAL pImg := 0    

    LOCAL nFaces
    LOCAL aFaces := {}
    LOCAL aFeatures := {}
    LOCAL nFPS := 0

    LOCAL nCos_score
    LOCAL nL2Score := 0

    LOCAL oDrFace

    DEFAULT lLandmark := .f.

   // oImage:Move(0,0,oImage:nWidth, oImage:nHeight, .t.)
   // oDlg:Move(0,0,oImage:nWidth, oImage:nHeight, .t.)
   // oDlg:Center()


    oDrFace := TDrFace():New()

    IF !oDrFace:LoadImage( cFile, @pImg )
        oDrFace:DeleteObject( pImg )
        RETURN NIL
    ENDIF    

    IF !oDrFace:DetectFaces( pImg, @nFaces, @aFaces, @aFeatures, @nFPS )
        oDrFace:DeleteObject( pImg )
       RETURN NIL
    ENDIF  

    oImage:bPainted := {| hDC | oDrFace:DrawFaces( hDC, aFaces, lLandmark ) }
    oImage:Refresh()

       
  //  nCos_score := oDrFace:MatchFaces(aFeatures[1][5], aFeatures[1][5], @nL2Score )

  //  ?nCos_score,nL2Score

    
   // ?aFeatures[1][1],aFeatures[1][2], aFeatures[1][3]
   // ?aFeatures[2][1],aFeatures[2][2], aFeatures[2][3]

    oDrFace:DeleteObject( pImg )


   
    
RETURN NIL


#include "fivewin.ch"

#define IMREAD_COLOR 1

CLASS TDrFace

    DATA oCv2
    DATA nFaces
    DATA aFaces
    DATA aFeatures

    METHOD New() CONSTRUCTOR
    METHOD LoadImage( cFile )
    METHOD DetectFaces( pImg, @nFaces, @aFaces, @aFeatures, @nFPS )

    METHOD DeleteObject( pImg ) INLINE ::oCv2:DeleteObject( pImg )
    METHOD DrawFaces( hDC, aFaces, lLandmark )
    METHOD MatchFaces( cFeature1, cFeature2, nL2Score )

    METHOD DrawRect( hDC, nTop, nLeft, nHeight, nWidth, nColor, nDepth ) 

ENDCLASS

METHOD New() CLASS TDrFace
    
    ::oCv2 := TCV2():New()

RETURN Self    


METHOD LoadImage( cFile, pImg ) CLASS TDrFace
          
    IF !::oCv2:imread( cFile, IMREAD_COLOR, @pImg )
       RETURN .F.
    ENDIF

RETURN .T.    

METHOD DetectFaces( pImg, nFaces, aFaces, aFeatures, nFPS ) CLASS TDrFace
    LOCAL lOk
    
    lOk := DetectFaces( pImg, @nFaces, @aFaces, @aFeatures )
    
    ::nFaces := nFaces
    ::aFaces := aFaces    
    
RETURN lOk


METHOD DrawFaces( hDC, aFaces, lLandmark ) CLASS TDrFace       
    LOCAL nI
    LOCAL nColor := CLR_BLUE    

    DEFAULT lLandmark := .f.

    FOR nI := 1 TO LEN( aFaces )
        ::DrawRect( hDC, aFaces[nI][2], aFaces[nI][1], aFaces[nI][4], aFaces[nI][3], CLR_HGREEN, 2 )
        IF lLandmark
                      
           nColor := CLR_HBLUE
           DrawCircle( hDC, aFaces[nI][5], aFaces[nI][6], 5, nColor )              
           DrawCircle( hDC, aFaces[nI][7], aFaces[nI][8], 5, nColor )
           
           nColor := CLR_HRED
           DrawCircle( hDC, aFaces[nI][9], aFaces[nI][10], 5, nColor )           

           nColor := CLR_HCYAN
           
           DrawCircle( hDC, aFaces[nI][11], aFaces[nI][12], 5, nColor )                      
           DrawCircle( hDC, aFaces[nI][13], aFaces[nI][14], 5, nColor )
           
        ENDIF
    NEXT

 RETURN NIL

 METHOD MatchFaces( cFeature1, cFeature2, nL2Score ) CLASS TDrFace       
    LOCAL nCos_score
                                                     
    nCos_score := MatchFaces( cFeature1, cFeature2, @nL2Score )
    
RETURN round( nCos_score, 8 )


METHOD DrawRect( hDC, nTop, nLeft, nHeight, nWidth, nColor, nDepth ) CLASS TDrFace     
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


FUNCTION DrawCircle( hDC, nX, nY, nWidth, nClr )
    LOCAL hPen   
    LOCAL hOldPen
    

    hPen    := CreatePen( PS_SOLID, nWidth, nClr )
    hOldPen := SelectObject( hDC, hPen )

    Ellipse( hDC, nX, nY,  nX + nWidth - 1, nY + nWidth - 1 )

    SelectObject ( hDC, hOldPen )
    DeleteObject( hPen )
    

RETURN NIL    

 



 
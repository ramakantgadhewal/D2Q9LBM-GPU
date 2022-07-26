#include <shape.hpp>

Shape::Shape(int nX, int nY)
{
    int nAll = nX * nY;
    shapeList = new bool [nAll];
    
    for (int i = 0; i < nAll; i++) {
        shapeList[i] = false;
    }
}

Shape::~Shape()
{
    delete shapeList;
}
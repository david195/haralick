/*
 * David Ruiz García. *
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

/*Statistical Properties of Co-occurrence Matrix*/

double Px(Mat matrix, int i){
  double result = 0.0;
  for(int j=0; j<matrix.size().width; j++){
    result += matrix.at<float>(Point(i,j));
  }
  return result;
}

double Py(Mat matrix, int j){
  double result = 0.0;
  for(int i=0; i<matrix.size().width; i++){
    result += matrix.at<float>(Point(i,j));
  }
  return result;
}

double Px_plus_y(Mat matrix, int k){
  double result = 0;
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      double aux1 = x+y;
      double aux2 = k * matrix.at<float>(Point(x,y));
      if(aux1 == aux2)
        result++;
    }
  }
  return result;
}

double Px_less_y(Mat matrix, int k){
  double result = 0;
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      double aux1 = abs(x-y);
      double aux2 = k * matrix.at<float>(Point(x,y));
      if(aux1 == aux2)
        result++;
    }
  }
  return result;
}

double HX(Mat matrix){
  double result = 0.0;
  for(int i=0; i<matrix.size().width; i++){
    double px = Px(matrix,i);
    if(px!=0)
      result += px * log(px);
  }
  return -result;
}

double HY(Mat matrix){
  double result = 0.0;
  for(int i=0; i<matrix.size().width; i++){
    double py = Py(matrix,i);
    if(py!=0)
      result += py * log(py);
  }
  return -result;
}

double HXY(Mat matrix){
  double result = 0.0;
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      if(matrix.at<float>(Point(x,y)) != 0)
      result += matrix.at<float>(Point(x,y)) * log(matrix.at<float>(Point(x,y)));
    }
  }
  return -result;
}

double HXY1(Mat matrix){
  double result = 0.0;
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      double px = Px(matrix,x);
      double py = Py(matrix,y);
      if(px !=0 && py !=0)
        result += matrix.at<float>(Point(x,y)) * log(px * py);
    }
  }
  return -result;
}

double HXY2(Mat matrix){
  double result = 0.0;
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      double px = Px(matrix,x);
      double py = Py(matrix,y);
      if(px !=0 && py !=0)
        result += px*py * log(px*py);
    }
  }
  return -result;
}

double Q(Mat matrix,int i, int j){
  double result = 0;
  for(int k=0; k<matrix.size().width; k++){
    result += ( matrix.at<float>(Point(i,k)) * matrix.at<float>(Point(i,k)) ) / ( Px(matrix,i) * Py(matrix,j) ) ;
  }
  return result;
}

/*Feature 1*/
double f1(Mat matrix){
  double feature=0.0;
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      double val = matrix.at<float>(Point(x,y));
      feature+=pow(val,2);
    }
  }
  return feature;
}

/*Feature 2*/
double f2(Mat matrix){
  double feature = 0.0;
  for(int n=0; n<255; n++){
    double value = 0.0;
    for(int x=0; x<matrix.size().width; x++){
      for(int y=0; y<matrix.size().height; y++){
        if(abs(x-y) == n)
          value += matrix.at<float>(Point(x,y));
      }
    }
    feature += pow(n,2) * value;
  }
  return feature;
}

/*Feature 3 - Correlation*/
double f3(Mat matrix){
  double feature = 0.0;
  double Mx=0.0, My=0.0;
  double Dx=0.0, Dy=0.0;
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      Mx += x * matrix.at<float>(Point(x,y));
      My += y * matrix.at<float>(Point(x,y));
    }
  }
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      Dx += pow((x - Mx),2) * matrix.at<float>(Point(x,y));
      Dy += pow((y - My),2) * matrix.at<float>(Point(x,y));
    }
  }
  Dx = sqrt(Dx);
  Dy = sqrt(Dy);
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      feature += matrix.at<float>(Point(x,y)) * ( ((x - Mx) * (y - My)) / (Dx * Dy) );
    }
  }

  return feature;
}

/*Feature 4 - Variance*/
double f4(Mat matrix){
  double feature = 0.0;
  Scalar med = mean(matrix);
  double median = double(med.val[0]);
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      feature += pow((x-median),2) * matrix.at<float>(Point(x,y));
    }
  }
  return feature;
}

/*Feature 5 - Inverse diference moment*/
double f5(Mat matrix){
  double feature = 0.0;
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      feature += 1 / (1 + pow((x-y),2)) * matrix.at<float>(Point(x,y));
    }
  }
  return feature;
}

/*Feature 6 - Sum Average*/
double f6(Mat matrix){
  double feature = 0.0;
  for(int i=2; i < matrix.size().width * 2; i++){
      feature += (i * Px_plus_y(matrix,i));
  }
  return feature;
}

/*Feature 8 - Sum Entropy*/
double f8(Mat matrix){
  double feature = 0.0;
  for(int i=2; i < matrix.size().width * 2; i++){
    double px_y = Px_plus_y(matrix,i);
    if(px_y!=0)
      feature += px_y * log(px_y);
  }
  return -feature;
}

/*Feature 7 - Sum Variance*/
double f7(Mat matrix){
  double feature = 0.0;
  double vf8 = f8(matrix);
  for(int i=2; i < matrix.size().width * 2; i++){
    double px_y = Px_plus_y(matrix,i);
    feature += pow((i-vf8),2) * px_y;
  }
  return feature;
}

/*Feature 9 - Entropy*/
double f9(Mat matrix){
  double feature = 0.0;
  for(int x=0; x<matrix.size().width; x++){
    for(int y=0; y<matrix.size().height; y++){
      if(matrix.at<float>(Point(x,y)) != 0)
        feature += matrix.at<float>(Point(x,y)) * log(matrix.at<float>(Point(x,y)));
    }
  }
  return -feature;
}

/*Feature 10 - diference Variance*/
double f10(Mat matrix){
  double feature = 0.0;
  for(int i=0; i<matrix.size().width-1; i++){
    feature += pow(i,2) * Px_less_y(matrix,i);
  }
  return feature;
}

/*Feature 11 - diference Entropy*/
double f11(Mat matrix){
  double feature = 0.0;
  for(int i=0; i < matrix.size().width; i++){
    double px_y = Px_less_y(matrix,i);
    if(px_y!=0)
      feature += px_y * log(px_y);
  }
  return -feature;
}

/*Feature 12 - info. Measure of Correlation 1*/
double f12(Mat matrix){
  double vHX = HX(matrix);
  double vHY = HY(matrix);
  double feature = HXY(matrix) - HXY1(matrix);
  if(vHX > vHY)
    feature = feature / vHX;
  else
    feature = feature / vHY;
  return feature;
}

/*Feature 13 - info. Measure of Correlation 2*/
double f13(Mat matrix){
  double feature = 1 - exp(-2 * (HXY2(matrix) - HXY(matrix)));
  feature = pow(feature,0.5);
  return feature;
}

double f14(Mat matrix){
  double feature = 0.0;
  double max=0;
  for(int i=0; i<matrix.size().width; i++){
    for(int j=0; j<matrix.size().height; j++){
      double Qval = Q(matrix,i,j);
      if(max<Qval){
        feature = max;
        max = Qval;
      }
    }
  }
  return sqrt(feature);
}

vector<double> haralick(Mat glcm){
  vector<double>features;
  /*Feature 1 - Angular second moment*/
  features.push_back(f1(glcm));
  /*Feature 2 - Contrast*/
  features.push_back(f2(glcm));
  /*Feature 3 - Correlation*/
  features.push_back(f3(glcm));
  /*Feature 4 - Sum of squares: Variance*/
  features.push_back(f4(glcm));
  /*Feature 5 - Inverse diference Moment*/
  features.push_back(f5(glcm));
  /*Feature 6 - Sum Average*/
  features.push_back(f6(glcm));
  /*Feature 7 - Sum Variance*/
  features.push_back(f7(glcm));
  /*Feature 8 - Sum Entropy*/
  features.push_back(f8(glcm));
  /*Feature 9 - Entropy*/
  features.push_back(f9(glcm));
  /*Feature 10 - DifVariance*/
  features.push_back(f10(glcm));
  /*Feature 11 - DifEntropy*/
  features.push_back(f11(glcm));
  /*Feature 12 - info. Measure of Correlation 1*/
  features.push_back(f12(glcm));
  /*Feature 13 - info. Measure of Correlation 2*/
  features.push_back(f13(glcm));
  /*Feature 14 - Max Correlation Coefficient*/
  features.push_back(f14(glcm));
  return features;
}

/*Calculation of Co-occurrence Matrix */
Mat GLCM(Mat src, int offset_x, int offset_y){
  double np = src.size().height*src.size().width;
  Mat matrix = Mat::zeros(256,256,CV_32FC1);
  for(int val1=0; val1<255; val1++){
    for(int val2=0; val2<255; val2++){
      int nc = 0;
      for(int x=0; x<src.size().width; x++){
        for(int y=0; y<src.size().height; y++){
          int intensity1 = src.at<uchar>(Point(x,y));
          int intensity2 = src.at<uchar>(Point(x+offset_x,y+offset_y));
          if(intensity1==val1 && intensity2==val2)
            nc++;
        }
      }
      double p = nc/np;
      matrix.at<float>(Point(val1,val2)) = p;
    }
  }
  return matrix;
}

int WriteCSV(vector<double> vec, string name, string dir){
  ostringstream row;
  row << name << ",";
  for(int i=0; i<vec.size()-1; i++){
    row << vec.at(i) << ",";
  }
  row << vec.at(vec.size()-1);
  ofstream file;
  file.open(dir, ios::app);
  if(!file) return -1;
  file << row.str() << endl;
  file.close();
  return 0;
}

int main( int argc, char** argv ){
  if(argc < 4){
    cout << "Insufficient parameters." << endl;
    return -1;
  }
  Mat src = imread( argv[1] ,CV_LOAD_IMAGE_GRAYSCALE);
  if( !src.data ){ cout << "Image not found" <<endl;return -1; }
  int offset_x,offset_y;
  int ang = stoi (argv[2],nullptr,10);
  int dis = stoi (argv[3],nullptr,10);
  switch (ang) {
    case 45:
      offset_y = dis;
      offset_x = dis;
      break;
    case 95:
      offset_x=0;
      offset_y=dis;
      break;
    case 135:
      offset_x = -dis;
      offset_y = -dis;
      break;
    default:
      offset_x=dis;
      offset_y=0;
    break;
  }
  if( offset_x == 0 && offset_y==0){
    cout << "both offset can not be 0" << endl;
    return -1;
  }
  Mat glcm_matrix = GLCM(src,offset_x,offset_y);

  vector<double>features = haralick(glcm_matrix);
  WriteCSV(features,argv[1],"out/feature_vectors.csv");
  for(int i=0; i<features.size(); i++){
    cout << i+1 << " : "<< features.at(i) << endl;
  }

  return 0;
}

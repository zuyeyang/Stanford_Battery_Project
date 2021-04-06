//
//  main.cpp
//  cs31_w1
//
//  Created by Yeyang Zu on 2021/4/5.
//

//file main.cpp
#include<iostream>
using namespace std;

int main(){
    double Pi=3.14;
    int r;
    int h;
    double v;
    cout << "Enter variale of r and h :";
    cin >> r >> h;
    cout << r << " " << h << endl; // helpful for debug by printing them out
    v = Pi * r * r * h;
    cout << "Volume is " << v<<endl;
}


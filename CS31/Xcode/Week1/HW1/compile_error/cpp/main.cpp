//
//  main.cpp
//  cpp
//
//  Created by Yeyang Zu on 2021/4/6.
//

// Code for Project 1
    // Report survey results
    
    #include <iostream>
    using namespace std;
    
    int main()
    {
        int numberSurveyed;
        int likeZoom;
        int sickOfZoom;
    
        cout << "How many students were surveyed? ";
        cin >> numberSurveyed;
        cout << "How many of them like having Zoom classes? ";
        cin >> likeZoom;
        cout << "How many of them are sick of Zoom? ";
        cin >> sickOfZoom;
    
        double pctLike = 100.0 * likeZoom / numberSurveyed;
        double pctSick = 100.0 * sickOfZoom / numberSurveyed;

        cout.setf(ios::fixed);
        cout.precision(1);
    
        cout << endl;
        cout << pctLike << "% like having Zoom classes." << endl;
        cout << pctSick << "% are sick of Zoom." << endl;

        if (likeZoom > sickOfZoom)
            cout << "More students like Zoom classes than are sick of Zoom." << endl;
        else
            cout << "More students are sick of Zoom than like Zoom classes." << endl;
    }

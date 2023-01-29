/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <queue>
#include <deque>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

// fixed sized buffer template
// will pop front if buffer is full and append a new image into the tail or end queue implementation
// overloading push function
template <typename T, int MaxLen, typename Container=std::deque<T>>
class FixedSizeQueue : public std::queue<T, Container> {
public:
    void push(const T& value) {
        if (this->size() == MaxLen) {
           this->c.pop_front(); 
        }
        std::queue<T, Container>::push(value);
    }
};

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    // Initializing ring buffer
    FixedSizeQueue<DataFrame,2> ring_buffer;
    /* MAIN LOOP OVER ALL IMAGES */
    std::vector<float> Keypointbefore;
    std::vector<float> KeypointAfter;
    std::vector<float> DescriptorMatches;
    std::vector<float> KeypointTime;
    std::vector<float> MatchingTime;
    float KeypointTime_ = 0.0;
    float MatchingTime_ = 0.0;
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
        
        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        ring_buffer.push(frame);
        //dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << "Image Index :"<< imgIndex <<endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = "HARRIS";

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, true,KeypointTime_,imgIndex);
        }
        else if(detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, true,KeypointTime_,imgIndex);
      
        }
        else if(detectorType.compare("FAST") == 0)
        {
            detKeypointsModern(keypoints, imgGray, "FAST",true,KeypointTime_,imgIndex);
        }
        else if(detectorType.compare("BRISK") == 0)
        {
            detKeypointsModern(keypoints, imgGray,"BRISK", true,KeypointTime_,imgIndex);
        }
        else if(detectorType.compare("ORB") == 0)
        {
            detKeypointsModern(keypoints, imgGray, "ORB", true,KeypointTime_,imgIndex);
        }
        else if(detectorType.compare("AKAZE") == 0)
        {
            detKeypointsModern(keypoints, imgGray, "AKAZE", true,KeypointTime_,imgIndex);
        }
        else if(detectorType.compare("SIFT") == 0)
        {
            detKeypointsModern(keypoints, imgGray, "SIFT", true,KeypointTime_,imgIndex);
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle
        cout << "#2 : DETECT KEYPOINTS before ROI done... Size:"<< keypoints.size() << endl;
        KeypointTime.push_back(KeypointTime_);
        Keypointbefore.push_back(keypoints.size());
        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        vector<cv::KeyPoint> keypointsPreceding; // create empty feature list for current image
        if (bFocusOnVehicle)
        {
            for(auto& key : keypoints)
            {
                bool result = vehicleRect.contains(key.pt);
                if(result == true)
                {
                    keypointsPreceding.push_back(key);
                }
            }
            keypoints = keypointsPreceding;
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        //(dataBuffer.end() - 1)->keypoints = keypoints;
        ring_buffer.back().keypoints = keypoints;
        KeypointAfter.push_back(keypoints.size());
        cout << "#2 : DETECT KEYPOINTS after ROI done... Size:"<< ring_buffer.back().keypoints.size() << endl;
    
        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
        cv::Mat descriptors;
        string descriptorType = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT, BRISK
        descKeypoints(ring_buffer.back().keypoints, ring_buffer.back().cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        //(dataBuffer.end() - 1)->descriptors = descriptors;
        ring_buffer.back().descriptors = descriptors;
        cout << "#3 : EXTRACT DESCRIPTORS done... "<< endl;

        if (ring_buffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors(ring_buffer.back().keypoints, ring_buffer.front().keypoints,
                             ring_buffer.back().descriptors, ring_buffer.front().descriptors,
                             matches, descriptorType, matcherType, selectorType,MatchingTime_);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            //(dataBuffer.end() - 1)->kptMatches = matches;
            ring_buffer.back().kptMatches = matches;
            
            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done / Size of Matches: " << (int)ring_buffer.back().kptMatches.size() << endl;
            
            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = (ring_buffer.back().cameraImg).clone();
                cv::drawMatches(ring_buffer.back().cameraImg, ring_buffer.back().keypoints,
                                ring_buffer.front().cameraImg, ring_buffer.front().keypoints,
                                ring_buffer.back().kptMatches , matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                string image_name= "Matcher Image #" + std::to_string((unsigned long)imgIndex)+".jpg";
                cv::imwrite(image_name,matchImg);
                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 10);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            
            bVis = false;
            cout << "---------------------------------------------------------------------------------------" << endl;
        }
        MatchingTime.push_back(MatchingTime_);
        DescriptorMatches.push_back((int)ring_buffer.back().kptMatches.size());

    } // eof loop over all images
            std::vector<std::pair<std::string, std::vector<float>>> vals= {{"Keypointbefore", Keypointbefore}, {"KeypointAfter", KeypointAfter}, {"DescriptorMatches", DescriptorMatches}, {"KeypointTime", KeypointTime}, {"MatchingTime", MatchingTime}};
            // Write the vector to CSV
            write_csv("SHITOMASI_SIFT.csv", vals);
        
    return 0;
}
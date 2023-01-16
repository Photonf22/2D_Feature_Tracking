#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        int k = 2;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        vector<cv::DMatch> bad_matches;
        double t2 = (double)cv::getTickCount();
        // TODO : implement k-nearest-neighbor matching
        matcher->knnMatch(descSource,descRef,knn_matches, 2);
        t2 = ((double)cv::getTickCount() - t2) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t2 / 1.0 << " ms" << endl;
        // TODO : filter matches using descriptor distance ratio test
        const float ratio_threshold = 0.8;
        for(size_t i = 0; i < knn_matches.size() ;i++)
        {
            if(knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance)
            {
                matches.push_back(knn_matches[i][0]);
            }
            else
            {
                bad_matches.push_back(knn_matches[i][0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    // Apply corner detection
    else if(descriptorType.compare("FAST") == 0)
    {
        int threshold_Fast = 10;
        bool non_maxSuppression = true;
        cv::FastFeatureDetector::DetectorType type_fast =cv:: FastFeatureDetector::TYPE_9_16;
        extractor = cv::FastFeatureDetector::create(threshold_Fast, non_maxSuppression, type_fast);
       
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor= cv::ORB::create();
        double t = (double)cv::getTickCount();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
        
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
std::vector<cv::KeyPoint> NMS_Algorithm(cv::Mat CornerHarris_img, int apertureSize, int threshold)
{
    // harris response matrix with all maximum pixels
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    std::vector<cv::KeyPoint> result;
    for(int i = 0;i < CornerHarris_img.rows ;i++)
    {
        for(int j = 0;j < CornerHarris_img.cols ;j++)
        {
            int pixel_intensity = (int)CornerHarris_img.at<float>(i,j);

            if(pixel_intensity > threshold)
            {
                cv::KeyPoint new_key;
                new_key.pt = cv::Point2f(i,j);
                new_key.size = 2*apertureSize;
                new_key.response = pixel_intensity;
            
            bool added = false;
            for(std::vector<cv::KeyPoint>::iterator iter= result.begin();iter != result.end();iter++)
            {  
                //check if overlap and % of overlap
                double percentage_overlap = cv::KeyPoint::overlap(new_key, *iter);
                if(percentage_overlap > maxOverlap)
                {
                    added = true;
                    // check which has more intensity/ response
                    if(new_key.response > iter->response)
                    {
                        *iter = new_key;
                        break;
                    }
                }
                
            }
            
            }
        }
    }
    return result;
}
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
        // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    keypoints = NMS_Algorithm(dst_norm_scaled,apertureSize,minResponse);
     // visualize keypoints
     // visualize results
    if (bVis)
    {
        string  windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // Apply corner detection
    if(detectorType.compare("FAST") == 0)
    {
        vector<cv::KeyPoint> kptsFAST;
        int threshold_Fast = 30;
        bool non_maxSuppression = true;
        cv::FastFeatureDetector::DetectorType type_fast =cv:: FastFeatureDetector::TYPE_9_16;
        double t = (double)cv::getTickCount();
        cv::FAST(img,kptsFAST,threshold_Fast, non_maxSuppression, type_fast);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if(detectorType.compare("BRISK") == 0)
    {
        int threshold_BRISK = 30;
        int octaves = 3;
        int patternScale = 1.0f;
        cv::Ptr<cv::BRISK> type_brisk = cv::BRISK::create();
        double t = (double)cv::getTickCount();
        type_brisk->detect(img, keypoints);
        
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if(detectorType.compare("ORB") == 0)
    {
        cv::Ptr<cv::FeatureDetector> type_ORB = cv::ORB::create();
        double t = (double)cv::getTickCount();
        type_ORB->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        cv::Ptr<cv::FeatureDetector> type_AKAZE = cv::AKAZE::create();
        double t = (double)cv::getTickCount();
        type_AKAZE->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if(detectorType.compare("SIFT") == 0)
    {
        cv::Ptr<cv::FeatureDetector> type_SIFT = cv::SIFT::create();
        double t = (double)cv::getTickCount();
        type_SIFT->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    
    // visualize keypoints
    // visualize results
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
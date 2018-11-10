//
// Created by hunglv on 09/11/2018.
//
#include <libfreenect.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
// init library of OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace cv;
using namespace std;


class myMutex {
	public:
		myMutex() {
			pthread_mutex_init( &m_mutex, NULL );
		}
		void lock() {
			pthread_mutex_lock( &m_mutex );
		}
		void unlock() {
			pthread_mutex_unlock( &m_mutex );
		}
	private:
		pthread_mutex_t m_mutex;
};


class MyFreenectDevice : public Freenect::FreenectDevice {
	public:
		MyFreenectDevice(freenect_context *_ctx, int _index)
	 		: Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT),
			m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(2048), m_new_rgb_frame(false),
			m_new_depth_frame(false), depthMat(Size(640,480),CV_16UC1),
			rgbMat(Size(640,480), CV_8UC3, Scalar(0)),
			ownMat(Size(640,480),CV_8UC3,Scalar(0)) {

			for( unsigned int i = 0 ; i < 2048 ; i++) {
				float v = i/2048.0;
				v = std::pow(v, 3)* 6;
				m_gamma[i] = v*6*256;
			}
		}

		// Do not call directly even in child
		void VideoCallback(void* _rgb, uint32_t timestamp) {
//			std::cout << "RGB callback" << std::endl;
			m_rgb_mutex.lock();
			uint8_t* rgb = static_cast<uint8_t*>(_rgb);
			rgbMat.data = rgb;
			m_new_rgb_frame = true;
			m_rgb_mutex.unlock();
		};

		// Do not call directly even in child
		void DepthCallback(void* _depth, uint32_t timestamp) {
//			std::cout << "Depth callback" << std::endl;
			m_depth_mutex.lock();
			uint16_t* depth = static_cast<uint16_t*>(_depth);
			depthMat.data = (uchar*) depth;
			m_new_depth_frame = true;
			m_depth_mutex.unlock();
		}

		bool getVideo(Mat& output) {
			m_rgb_mutex.lock();
			if(m_new_rgb_frame) {
				cv::cvtColor(rgbMat, output, COLOR_RGB2GRAY);
				m_new_rgb_frame = false;
				m_rgb_mutex.unlock();
				return true;
			} else {
				m_rgb_mutex.unlock();
				return false;
			}
		}

		bool getDepth(Mat& output) {
				m_depth_mutex.lock();
				if(m_new_depth_frame) {
					depthMat.copyTo(output);
					m_new_depth_frame = false;
					m_depth_mutex.unlock();
					return true;
				} else {
					m_depth_mutex.unlock();
					return false;
				}
			}
	private:
		std::vector<uint8_t> m_buffer_depth;
		std::vector<uint8_t> m_buffer_rgb;
		std::vector<uint16_t> m_gamma;
		Mat depthMat;
		Mat rgbMat;
		Mat ownMat;
		myMutex m_rgb_mutex;
		myMutex m_depth_mutex;
		bool m_new_rgb_frame;
		bool m_new_depth_frame;
};

cv::Mat smoothDepth(cv::Mat src){
    const uchar noDepth = 255;
    cv::Mat temp, temp2, small_depth, dst;
    src.copyTo(dst);
    cv::resize(dst, small_depth, cv::Size(), 0.2, 0.2);
    cv::inpaint(small_depth, (small_depth==noDepth), temp, 5.0, cv::INPAINT_TELEA);
    cv::resize(temp, temp2, dst.size());
    temp2.copyTo(dst, (dst==noDepth));
    return dst;
}

cv::Mat colorDepth(cv::Mat src){
    const uchar minDepth = 0, maxDepth = 100;
    const uchar initR = 255, initG = 255, initB = 255;
    Mat rgb_depth(src.size(),CV_8UC3);
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            uchar depth = src.at<uchar>(i,j);
            uchar intensity = depth;
            uchar newR = (uchar)((int)initR * (int)intensity / 255);
            uchar newG = (uchar)((int)initG * (int)intensity / 255);
            uchar newB = (uchar)((int)initB * (int)intensity / 255);
//            std::cout << intensity << " " << newR << " " << newG << " " << newB << std::endl;
//            rgb_depth.at<Vec3b>(i,j) = Vec3b(initR*intensity/255, initG*intensity/255, initB*intensity/255);
            rgb_depth.at<Vec3b>(i,j) = Vec3b(newR, newG, newB);
        }
    }
    return rgb_depth;
}

int main(int argc, char **argv) {
	bool die(false);
	string filename("snapshot");
	string suffix(".png");
	int i_snap(0),iter(0);

	Mat depthMat(Size(640,480),CV_16UC1);
	Mat depthf (Size(640,480),CV_8UC1);
	Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
	Mat ownMat(Size(640,480),CV_8UC3,Scalar(0));

	// The next two lines must be changed as Freenect::Freenect
	// isn't a template but the method createDevice:
	// Freenect::Freenect<MyFreenectDevice> freenect;
	// MyFreenectDevice& device = freenect.createDevice(0);
	// by these two lines:

	Freenect::Freenect freenect;
	MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);
	namedWindow("rgb",1);
	device.startVideo();
	device.startDepth();
    int frame_width = 640;
    int frame_height = 480;
    cv::VideoWriter video_1("output_rgb.avi",cv::VideoWriter::fourcc('X','V','I','D'),10, cv::Size(frame_width,frame_height));
    cv::VideoWriter video_2("output_depth.avi",cv::VideoWriter::fourcc('X','V','I','D'),10, cv::Size(frame_width,frame_height));

    while (1) {
        device.getVideo(rgbMat);
        device.getDepth(depthMat);
        cv::cvtColor(depthMat, depthMat, cv::COLOR_GRAY2BGR);
        cv::imshow("rgb", rgbMat);
        cv::imshow("depth", depthMat);
//        cv::cvtColor(rgbMat, rgbMat, cv::COLOR_GRAY2BGR);
//        cv::imshow("rgb", rgbMat);
//        depthMat.convertTo(depthMat, CV_8UC1, 255.0/2048.0);
//        cv::Mat smooth = smoothDepth(depthMat);
//        cv::cvtColor(smooth, smooth, cv::COLOR_GRAY2BGR);
//        cv::imshow("smooth-depth", smooth);
        int k = cv::waitKey(30);
        video_1.write(rgbMat);
        video_2.write(depthMat);
        if (k == 27) {
//            std::cout << total.rows << " " << total.cols << " " << total.size() << std::endl;
            break;
        }
//		cv::imshow("rgb", rgbMat);
//		depthMat.convertTo(depthf, CV_8UC1, 255.0/2048.0);
////		cv::imshow("depth",color_depthf);
//
//        cv::Mat smooth = smoothDepth(depthf);
//        cv::imshow("smooth-depth", smooth);
//
//        cv::Mat color_depth;
//        cv::applyColorMap(smooth*1.5, color_depth, COLORMAP_JET);
//        cv::imshow("color-depth", color_depth);
////        cv::Mat edge_depth;
////        cv::Canny(smooth_depth, edge_depth, 20, 60, 3);
////        cv::imshow("edge-depth", edge_depth);
//
//		char k = cv::waitKey(5);
//		if( k == 27 ){
////			cvDestroyWindow("rgb");
////			cvDestroyWindow("depth");
//			break;
//		}
//		if( k == 8 ) {
//			std::ostringstream file;
//			file << filename << i_snap << suffix;
//			cv::imwrite(file.str(),rgbMat);
//			i_snap++;
//		}
////		if(iter >= 1000) break;
////		iter++;
    }

    video_1.release();
	video_2.release();


	device.stopVideo();
	device.stopDepth();

	return 0;
}

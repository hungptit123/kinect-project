#include <cstdlib>
#include <iostream>
#include <vector>
#include <pthread.h>
#include <libfreenect.hpp>
// init library of OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

// init library of PCL
#include <pcl/console/parse.h>
#include <pcl/cloud_iterator.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/PCLImage.h>
#include <pcl/PointIndices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <boost/thread/thread.hpp>

//init library of OpenGL
#include <GL/glew.h>
#include <GL/glxew.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <cmath>

class Mutex
{
public:
    Mutex()
    {
        pthread_mutex_init(&m_mutex, NULL);
    }

    void lock()
    {
        pthread_mutex_lock(&m_mutex);
    }

    void unlock()
    {
        pthread_mutex_unlock(&m_mutex);
    }

    class ScopedLock
    {
    public:
        ScopedLock(Mutex &mutex) : _mutex(mutex)
        {
            _mutex.lock();
        }

        ~ScopedLock()
        {
            _mutex.unlock();
        }

    private:
        Mutex &_mutex;
    };

private:
    pthread_mutex_t m_mutex;
};


class MyFreenectDevice : public Freenect::FreenectDevice
{
public:
    MyFreenectDevice(freenect_context *_ctx, int _index)
            : Freenect::FreenectDevice(_ctx, _index),
              m_buffer_video(freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes),
              m_buffer_depth(freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED).bytes / 2),
              m_new_rgb_frame(false), m_new_depth_frame(false)
    {
        setDepthFormat(FREENECT_DEPTH_REGISTERED);
    }

    // Do not call directly, even in child
    void VideoCallback(void *_rgb, uint32_t timestamp)
    {
        Mutex::ScopedLock lock(m_rgb_mutex);
        uint8_t* rgb = static_cast<uint8_t*>(_rgb);
        copy(rgb, rgb+getVideoBufferSize(), m_buffer_video.begin());
        m_new_rgb_frame = true;
    }

    // Do not call directly, even in child
    void DepthCallback(void *_depth, uint32_t timestamp)
    {
        Mutex::ScopedLock lock(m_depth_mutex);
        uint16_t* depth = static_cast<uint16_t*>(_depth);
        copy(depth, depth+getDepthBufferSize()/2, m_buffer_depth.begin());
        m_new_depth_frame = true;
    }

    bool getRGB(std::vector<uint8_t> &buffer)
    {
        Mutex::ScopedLock lock(m_rgb_mutex);

        if (!m_new_rgb_frame)
            return false;

        buffer.swap(m_buffer_video);
        m_new_rgb_frame = false;

        return true;
    }

    bool getDepth(std::vector<uint16_t> &buffer)
    {
        Mutex::ScopedLock lock(m_depth_mutex);

        if (!m_new_depth_frame)
            return false;

        buffer.swap(m_buffer_depth);
        m_new_depth_frame = false;

        return true;
    }

private:
    Mutex m_rgb_mutex;
    Mutex m_depth_mutex;
    std::vector<uint8_t> m_buffer_video;
    std::vector<uint16_t> m_buffer_depth;
    bool m_new_rgb_frame;
    bool m_new_depth_frame;
};


Freenect::Freenect freenect;
MyFreenectDevice* device;

int window(0);                // Glut window identifier
int mx = -1, my = -1;         // Prevous mouse coordinates
float anglex = 0, angley = 0; // Panning angles
float zoom = 1;               // Zoom factor
bool color = true;            // Flag to indicate to use of color in the cloud

void GetCloudFromDevice(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud){
    static std::vector<uint8_t> rgb(640*480*3);
    static std::vector<uint16_t> depth(640*480);

    device->getRGB(rgb);
    device->getDepth(depth);

    float f = 595.f;
//        depth_image = cv::imread("DEPTH_IMAGE.jpg", cv::COLOR_GRAY2BGR);
    cloud->width = 640;
    cloud->height = 480;
    cloud->points.resize(cloud->width*cloud->height);
    cloud->is_dense = false;
    for(int i = 0; i < cloud->width*cloud->height; i++){
        cloud->points[i].r = rgb[3*i+0];
        cloud->points[i].g = rgb[3*i+1];
        cloud->points[i].b = rgb[3*i+2];
        cloud->points[i].x = (i%cloud->width - (cloud->width-1)/2.f) * depth[i] / f;
        cloud->points[i].y = (i/cloud->width - (cloud->height-1)/2.f) * depth[i] / f;
        cloud->points[i].z = depth[i];
        if(depth[i]!=0) std::cout << depth[i] << " ";
    }
    std::cout << std::endl;
}

void ProcessRANSAC(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &final){
    std::vector<int> inliers;
    pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr
            model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cloud));

    pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_p);
    ransac.setDistanceThreshold (10.0);
    ransac.computeModel();
    ransac.getInliers(inliers);

    pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud, inliers, *final);
}

void DisplayCloudXYZRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
    glPointSize(1.0f);
    glBegin(GL_POINTS);
    if (!color) glColor3ub(255, 255, 255);
    for (int i = 0; i < cloud->width*cloud->height; ++i)
    {
        if (color)
            glColor3ub(cloud->points[i].r, cloud->points[i].g, cloud->points[i].b);
        glVertex3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
    }
    glEnd();
}

void DrawGLScene()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    GetCloudFromDevice(cloud);

//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ransac (new pcl::PointCloud<pcl::PointXYZRGB>);
//    ProcessRANSAC(cloud, cloud_ransac);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    DisplayCloudXYZRGB(cloud);

    // Draw the world coordinate frame
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3ub(255, 0, 0);  // X-axis
    glVertex3f(  0, 0, 0);
    glVertex3f( 50, 0, 0);
    glColor3ub(0, 255, 0);  // Y-axis
    glVertex3f(0,   0, 0);
    glVertex3f(0,  50, 0);
    glColor3ub(0, 0, 255);  // Z-axis
    glVertex3f(0, 0,   0);
    glVertex3f(0, 0,  50);
    glEnd();

    // Place the camera
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(zoom, zoom, 1);
    gluLookAt( -7*anglex, -7*angley, -1000.0,
               0.0,       0.0,  2000.0,
               0.0,      -1.0,     0.0 );

    glutSwapBuffers();
}


void keyPressed(unsigned char key, int x, int y)
{
    switch (key)
    {
        case  'C':
        case  'c':
            color = !color;
            break;

        case  'Q':
        case  'q':
        case 0x1B:  // ESC
            glutDestroyWindow(window);
            device->stopDepth();
            device->stopVideo();
            exit(0);
    }
}


void mouseMoved(int x, int y)
{
    if (mx >= 0 && my >= 0)
    {
        anglex += x - mx;
        angley += y - my;
    }

    mx = x;
    my = y;
}


void mouseButtonPressed(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        switch (button)
        {
            case GLUT_LEFT_BUTTON:
                mx = x;
                my = y;
                break;

            case 3:
                zoom *= 1.2f;
                break;

            case 4:
                zoom /= 1.2f;
                break;
        }
    }
    else if (state == GLUT_UP && button == GLUT_LEFT_BUTTON)
    {
        mx = -1;
        my = -1;
    }
}


void resizeGLScene(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(50.0, (float)width / height, 900.0, 11000.0);

    glMatrixMode(GL_MODELVIEW);
}


void idleGLScene()
{
    glutPostRedisplay();
}


void printInfo()
{
    std::cout << "\nAvailable Controls:"              << std::endl;
    std::cout << "==================="                << std::endl;
    std::cout << "Rotate       :   Mouse Left Button" << std::endl;
    std::cout << "Zoom         :   Mouse Wheel"       << std::endl;
    std::cout << "Toggle Color :   C"                 << std::endl;
    std::cout << "Quit         :   Q or Esc\n"        << std::endl;
}

int main(int argc, char** argv)  {
//    device = &freenect.createDevice<MyFreenectDevice>(0);
//    device->startVideo();
//    device->startDepth();
//
//    device->getRGB()


    cv::VideoCapture cap;
    cap = cv::VideoCapture("output.avi");
//    cv::Mat image;
//    image = cv::imread("DEMPTH_IMAGE.jpg", cv::COLOR_RGB2GRAY);
    while(cap.isOpened()) {
        cv::Mat3f frame_rgb_depth(480,640*2,CV_32FC3);
        cap >> frame_rgb_depth;
        cv::Rect roi;
        roi.width = frame_rgb_depth.size().width/2;
        roi.height = frame_rgb_depth.size().height;

        cv::Mat3f rgb = frame_rgb_depth(roi);
        cv::imshow("frame_rgb", rgb);

        roi.x = 640;
        roi.y = 0;

        cv::Mat3f depth(480, 640, CV_32FC3);
        depth = frame_rgb_depth(roi);
        cv::Mat3f normals(depth.rows, depth.cols, CV_32FC3);

//        cv::cvtColor(depth, depth, cv::COLOR_RGB2GRAY);
//        depth.convertTo(depth, CV_32FC3);
//        cv::imwrite("depth.jpg", depth);
        cv::imshow("depth", depth);
//        std::cout << depth.at<cv::Vec3f>(0,0) << " ";
//        cv::Mat3f normals(de);
//        depth.copyTo(normals);
        for(int x = 0; x < depth.rows; x++) {
            for (int y = 0; y < depth.cols; y++) {
                float dzdx = 0.0, dzdy = 0.0;
                if(x == 0 || x == depth.rows-1){
                    dzdx = (depth.at<float>(x, y) + depth.at<float>(x, y)) / 2.0;
                } else {
                    dzdx = (depth.at<float>(x + 1, y) + depth.at<float>(x - 1, y)) / 2.0;
                }
                if(y == 0 || y == depth.rows-1){
                    dzdy = (depth.at<float>(x, y) + depth.at<float>(x, y)) / 2.0 ;
                } else {
                    dzdy = (depth.at<float>(x, y + 1) + depth.at<float>(x, y - 1)) / 2.0;
                }
                cv::Vec3f d(-dzdx, -dzdy, 1.0f);
                cv::Vec3f n = normalize(d);
                normals.at<cv::Vec3f>(x, y) = n;

//                if(x!=0 and y!=0 and x!=depth.rows-1 and y != depth.cols-1) {
//                    normals.at<cv::Vec3b>(x, y) = (depth.at<cv::Vec3b>(x-1, y)+depth.at<cv::Vec3b>(x+1, y))/2;
//                } else normals.at<cv::Vec3b>(x, y) = depth.at<cv::Vec3b>(x, y);
            }
        }
////        normals.convertTo(normals, CV_32FC3);
//        cv::resize(normals, normals, cv::Size(), 640.0/480.0, 1.0);
        cv::imshow("normal", normals);
//        cv::imwrite("normal.jpg", normals);
//        cv::imshow("depth", depth);


        //        cv::cvtColor(frame_rgb_depth, frame_rgb, cv::COLOR_RGB2GRAY);
//        int a = frame_rgb.at<cv::Vec3b>(0,0)[1];
        int k = cv::waitKey(30);

        if (k==27) {
            break;
        }
    }
//    glutInit(&argc, argv);
//
//    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
//    glutInitWindowSize(640, 480);
//    glutInitWindowPosition(0, 0);
//
//    window = glutCreateWindow("LibFreenect");
//    glClearColor(0.45f, 0.45f, 0.45f, 0.0f);
//
//    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_ALPHA_TEST);
//    glAlphaFunc(GL_GREATER, 0.0f);
//
//    glMatrixMode(GL_PROJECTION);
//    gluPerspective(50.0, 1.0, 900.0, 11000.0);
//
//    glutDisplayFunc(&DrawGLScene);
//    glutIdleFunc(&idleGLScene);
//    glutReshapeFunc(&resizeGLScene);
//    glutKeyboardFunc(&keyPressed);
//    glutMotionFunc(&mouseMoved);
//    glutMouseFunc(&mouseButtonPressed);
//
//    printInfo();
//
//    glutMainLoop();

    return 0;
}

///*
// * we need to 3 function
// *
// */
//
////function draw
//void RenderScene() {
//    glutWireTeapot(2.0);
//        glFlush();
//}
//
//void Reshape(int width, int height) {
//    glViewport(0,0, width, height);
//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//
//    glOrtho(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0);
//}
//
//void Init() {
//    glClearColor(0.0, 0.0, 0.0, 0.0); // delete window into black
//}

//int main(int argc, char **argv) {
//    glutInit(&argc, argv);
//    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
//    glutInitWindowSize(500,500); // size window display
//    glutInitWindowPosition(100,100); // position display window
//    glutCreateWindow("opengl");
//
//    Init();
//    glutReshapeFunc(Reshape);
//    glutDisplayFunc(RenderScene);
//    glutMainLoop();
//}


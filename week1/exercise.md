# Exercise: Turtlebot / ROS / Matlab infrastructure

**Objective**: To get the network infrastructure between your computer and Turtlebot up and running, and to get MatLab communicating with the TurtleBot.

Each TurtleBot3 - Burger holds:
* A Raspberry pi
* An OpenCR (Open Source Control Module for ROS), which contains motor controllers and IMU

The Raspberry pi is configured with:
* Ubuntu 20.04 LTS. Username is "ubuntu" password is “turtlebot”
* ROS Noetic

## Setting up the Raspberry Pi on the Turtlebot
The Raspberry Pi has already been set-up with ROS, but you need to set-up a connection between your computer and the Turtlebot.

## 1. Network setup
### WiFi
You will control the Turtlebot 3 wirelessly through WiFi. First, you need to configure a wireless hotspot on another device -- a dedicated router or your cell phone -- throught which you can communicate with the Turtlebot.

Out-of-the-box the turtlebot will connect to a hotspot with name ```turtlebot```and password ```turtlebot3```. Change these so it can connect to your own hostpot following the steps below.

* 1.a setup a hotspot and note the SSID and password
Turn on network sharing on your cell phone or router and connect your computer to that hotspot.

* 1.b Setup the Raspberry Pi to connect to that SSID
You should edit the following file on the Raspberry Pi and replace the existing SSID and Password with the one noted.
```
/etc/netplan/50-cloud-init.yaml
```
You can edit the file either though SSH, if you can access the Raspberry Pi through its existing network connection. Note that indentation is important in the YAML-file.

Restart the Turtlebot and wait for it to connect to your hotspot. On your hotspot, note the IP-address assigned to the Turlebot.

### Ethernet
It is also possible to access the robot via the ethernet cable. For this you need to setup static IPs for both your laptop and the robot.

## 2. Turtlebot
The TurtleBot is built according to this description: https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/#overview 
The Turtlebot 3 Burger comes without cameras, but we attach cameras, which will be used in future exercies.

## 3. Starting up the turtlebot

For running the examples below, your TurtleBot should be started by running the following command on your turtlebot:

From your computer SSH into the Turtlebot using it's ip (e.g. 192.168.1.200). Username is `ubuntu`, password is `turtlebot`
```
ssh ubuntu@192.168.1.200
```

Next you need to install the lidar driver. Check for the right version, either for lds-01:
```
sudo apt install libudev-dev
cd ~/catkin_ws/src/
git clone -b develop https://github.com/ROBOTIS-GIT/hls_lfcd_lds_driver.git
echo ’export LDS_MODEL=LDS-01’ >> ~/.bashrc
source ~/.bashrc
cd ~/catkin_ws/
catkin_make
source devel/setup.bash
```

or [lds-02](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#new-lds-02-configuration-2).

If catkin_make gets stuck, kill it with ctrl-c and run it again. Do this a few times if needed.

Setup the ROS HOSTNAME and ROS MASTER URI so that the master is visible from another machine in the same network:

```
export ROS_HOSTNAME=192.168.1.200
export ROS_MASTER_URI=http://192.168.1.200:33849
```

start-up the turtlebot by running:
```
roslaunch turtlebot3_bringup turtlebot3_robot.launch
```
If you get plenty of errors, the opencr might need to be updated: https://emanual.robotis.com/docs/en/platform/turtlebot3/opencr_setup/, make sure to select the NOETIC version.

You can also start-up the camera by running
```
roslaunch turtlebot3_bringup turtlebot3_rpicamera.launch
```
but we will not use that sensor in this exercise.



## 4. MatLab installation
Initialization inside MatLab is recommended as:
```
% assuming the Turtlebot ip is 192.168.1.200
setenv('ROS_MASTER_URI','http://192.168.1.200:11311')
% assuming your own ip is 192.168.1.100
setenv('ROS_IP','192.168.1.100')
rosinit('http://192.168.1.200:11311','NodeHost','192.168.1.100');
```

If you are using Windows, you might have to disable your firewall in order to have MatLab communicating with the TurtleBot.


### Examples for testing:													
* Getting Started with a Real TurtleBot, [https://se.mathworks.com/help/robotics/examples/get-started-with-a-real-turtlebot.html](https://se.mathworks.com/help/robotics/examples/get-started-with-a-real-turtlebot.html) Note that the environment variables (ROS_IP and ROS_MASTER_URI) could be set inside the `.bashrc` to aovid doing this each time. The printenv command can be used to verify that the parameters are set.
* Communicate with the TurtleBot,
[https://se.mathworks.com/help/robotics/examples/communicate-with-the-turtlebot.html](https://se.mathworks.com/help/robotics/examples/communicate-with-the-turtlebot.html) - Stop just before the section “Receive Image Data”
Make the robot follow a simple geometric trajectory, e.g.:

![path](path.svg)

## Some hints:
Publisher for the robot's velocity and a message for that topic:
```
robot = rospublisher('/cmd_vel');
velmsg = rosmessage(robot);
```

Define parameters for either Linear.X / Angular.Z (either one or both):
```
velmsg.Angular.Z = 0.6;	% Angular velocity (rad/s)
velmsg.Linear.X = 0.1; % Linear velocity (m/s), +forward,-reverse
```

Note that the complete velmsg is sent to the robot. If the robot has to move straight forward then set the
Angular.Z parameter to 0.
Update robot with actual velocity message:
```
send(robot,velmsg);
```

Make use of odometry topic to verify “Stop” position. Measure the robots accuracy with real
measurements. Repeat the experiment a number of times – is the accuracy similar for all experiments?
Note that you might reset the odometry inside the TurtleBot.



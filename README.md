This is a sample application that demonstrates how to integrate the KudanCV library on desktop platforms such as macOS and Windows.



## Getting the project to compile
----
In order to get this sample project to compile, you will need to download the OpenCV libraries for your platform and make sure to link against them in your IDE.

This Sample Demo has been tested against **_OpenCV 3.0 on macOS_** and **_OpenCV 3.1 on Windows_**. Using other versions of OpenCV has not been tested and therefore is not supported.

In this sample project, OpenCV is used for obtaining the camera feed and subsequently drawing this to the screen.

**You do not need to use OpenCV in your own applications. The KudanCV library can process camera inputs from any source.**

After you have been sent the KudanCV library, just drop it and the `KudanCV.h` file into the `KudanCV Demo` folder that is in the root of this repository.



## Getting the tracker to run
----
Once your project is compiling, the next step is to ensure that it actually runs without errors.

In order to do this have `main.cpp` open in your IDE. On line `23`, you will find `kLicenseKey` - assign to this variable the license key that you have received from Kudan. **Make sure you do not publicly share this license key.**

Next, on line `26` you will find `kMarkerPath`.  Here you will need to assign the full path of the `lego.jpg` marker, which is included in this repository.

At this point, your project should compile and run. If the library encounters any issues it will throw an exception to let you know what is missing or possibly going wrong.



## Contact
----
If you have any questions, please visit our [support area](https://www.kudan.eu/sdk-support/) or alternatively visit our [forums](https://forum.kudan.eu).
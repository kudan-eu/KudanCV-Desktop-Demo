This is a sample application that demonstrates how to integrate the KudanCV library on desktop platforms such as macOS and Windows.



## Getting the project to compile
----
In order to get this sample project to compile, you will need to download the OpenCV libraries for your platform and make sure to link against them in your IDE.

**macOS users**:
If your OpenCV files reside in `/usr/local/` they should be picked up by default. Ensure to change the `Header Search Paths` and `Library Search Paths` in Xcode.

**Windows users**:  
The Visual Studio project is setup to look for the OpenCV headers and library in `C:/Development/opencv/`. Please change this as appropriate.

----

This Sample Demo has been tested against **_OpenCV 3.0 on macOS_** and **_OpenCV 3.1 on Windows_**. Using other versions of OpenCV has not been tested and therefore is not supported.

In this sample project, OpenCV is used for obtaining the camera feed and subsequently drawing this to the screen.

**You do not need to use OpenCV in your own applications. The KudanCV library can process camera inputs from any source.**

After you have been sent the KudanCV library, just drop in the static library and the `KudanCV.h` file into the `KudanCV Demo` folder that is in the root of this repository.



## Getting the sample to run
----
Once your project is compiling, the next step is to ensure that it actually runs without errors.

In order to do this make the following changes in `main.cpp`.  
On line `23`, you will find `kLicenseKey` - assign to this variable the license key that you have received from Kudan. **Make sure you do not publicly share this license key.**

On line `26`, you will find `kMarkerPath` - assign the full path of the `lego.jpg` marker, which is included in this repository.

At this point, your project should compile and run. If the library encounters any issues it will throw an exception to let you know what is missing or possibly going wrong.

#### Special note for Windows users:
Once your project has compiled in Visual Studio, you may find that you need to copy across your OpenCV dll and the cURL library (which was in the Kudan package) into the folder which contains your compiled `.exe`. Otherwise the executable will fail to launch citing that it is missing the OpenCV and cURL libraries.


## Contact
----
If you have any questions, please visit our [support area](https://www.kudan.eu/sdk-support/) or alternatively visit our [forums](https://forum.kudan.eu).
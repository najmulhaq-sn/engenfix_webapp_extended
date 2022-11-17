
# EngenFix

**Automated Vehicle Damage Recognition & Assessment Web Application**


<br />



<br />

[![](https://img.shields.io/badge/Made_with-Python_3.8-blue.svg)]()
[![](https://img.shields.io/badge/App-FastAPI-important.svg)]()
[![](https://img.shields.io/badge/Detectron2-0.6-green.svg)]()
[![](https://img.shields.io/badge/torch-1.12.0-red.svg)]()
[![](https://img.shields.io/badge/Powered_by-Engenuity_Ai-yellow.svg)]()
[![](https://img.shields.io/badge/Product-EngenFix-1f425f.svg)]()
<!-- [![](https://img.shields.io/badge/torch-1.11.0-red.svg)]()
[![](https://img.shields.io/badge/Made_with-Flask-important.svg)]() -->

<br />


## Setting up the App Locally

Here is the guide for app installation in your own environments.


### Prerequisites

- Download all the Damage Detection & Segmentation Models.

    - [Broken Class Model](https://engenuitylk.sharepoint.com/:f:/s/PeoplesInsurance/EogGK6K0mKtKpDtAak3a-JwB4AWr1LJAP-abSImZz__GkA?e=r6Knom)
    - [Scratch Class Model](https://engenuitylk.sharepoint.com/:f:/s/PeoplesInsurance/EtOCj1T9-1FLk8hSgzH9MiABxpznh2qeWOh4ZA5gDFhPHw?e=cr6x5V)
    - [Dent Class Model](https://engenuitylk.sharepoint.com/:f:/s/PeoplesInsurance/EovlKYrviL9JsqxlEuQsPqkBFAdNtD_QQzdgBag56YMaaA?e=XB0Zj2)
    - [Glass Shatter Class Model](https://engenuitylk.sharepoint.com/:f:/s/PeoplesInsurance/Ekm62rbe1KFKmN4YkXn2eUMBafKN6Vs4ktt2str_rFkLJw?e=yU6JoB)

- Download all the Classification Models
    - [Damage Severity Model](https://engenuitylk.sharepoint.com/:u:/s/PeoplesInsurance/EWCDS95Vd2pCvBasoAAWC88BTjyt0D1_xL8C0UozjvdYvA?e=cHg1WJ)
    - [Vehicle Orientation Model](https://engenuitylk.sharepoint.com/:u:/s/PeoplesInsurance/ETOliLPFBWJCp2dUM4jbv5MBP_VDt2IanBi_QMmVr7erkg?e=eYrPhM)
    - [Vehicle Type Model](https://engenuitylk.sharepoint.com/:u:/s/PeoplesInsurance/ESlmzXINoqBOk2so2PqX4JIBp5FEIhzfheVzBickX4-lkg?e=2DbJpE)
    - [Vehicle Validation Model](https://engenuitylk.sharepoint.com/:u:/s/PeoplesInsurance/EU7eJhahivRNup10m5eM86sB9JflyRhIsUP1JWKu28P5rg?e=TSgCFE)

### Setup a New Virtual Environment (Optional)

- Create a new virtual environment

    ```bash
    $ conda create --name <env-name> python=3.8
    $ # specify a convienient name for <env-name> as for the new env
    ```

- To activate the created environment

    ```bash
    $ conda activate <env-name>
    $ # replace the specified name with <env-name>
    ```

### Installation

Follow the below steps for the installation.

- Clone the repository:

    ```bash
    $ https://github.com/engenuityai/engenfix-backend-api.git
    ```

    or download the zip file from the [repository](https://github.com/engenuityai/engenfix-backend-api.git) & extract it.

    Then navigate to the project root folder.


    ```bash
    $ cd engenfix-backend-api
    ```

- Copy the Downloaded Damage Class Models and place inside the `model` folder.

- Install the Requirements: 

    ```bash
    $ pip install -r requirements.txt
    ```

    To install the Detectron2, execute the below:

    ```bash
    pip install git+https://github.com/facebookresearch/detectron2
    ```

- Run the App on localhost:

    ```bash
    $ uvicorn main:app --reload
    $ # You can change to any specific host and port to serve the app
    ```

-  App will be Running at: 
<http://127.0.0.1:8000>



<br />

## ✨ Code-base structure

The project code base structure is as below:

```bash
< PROJECT ROOT >
   |
   |-- assets/                                  # Folder to store project io files
   |    |-- input/                              # Contains input images
   |    |-- output/                             # Contains final outputs
   |     
   |-- model/                                   # Model files
   |    |-- ver2/                               # Current version of the model
   |        |-- broken_model.pth                # Broken class model file
   |        |-- broken_model.yaml               # Broken model configuration file
   |        |-- dent_model.pth                  # Dent class model file
   |        |-- dent_model.yaml                 # Dent model configuration file
   |        |-- glass_model.pth                 # Glass Shatter class model file
   |        |-- glass_model.yaml                # Glass Shatter model configuration file
   |        |-- scratch_model.pth               # Scratch class model file
   |        |-- scratch_model.yaml              # Scratch model configuration file
   |
   |
   |-- utils/                                   # Support Functions
   |    |-- __init__.py                         # Module initialization
   |    |-- cnt_draw.py                         # Draw contours/masks on the image
   |    |-- det_models.py                       # Model inference class
   |
   |
   |-- unit_test.py                             # Unit test file
   |
   |-- requirements.txt                         # Requirements file
   |-- Dockerfile                               # Dockerfile Script
   |
   |-- main.py                                  # Main App Starter - ASGI gateway
   |
   |-- ************************************************************************
```

<br />

## More Information

- [Documentation](https://engenuitylk.sharepoint.com/:w:/s/PeoplesInsurance/EWRhEEfmiFRLlROaQUoz23AB1-AzkUQRPdLuGc7YWgziHw?e=t8uOkD) with more details.

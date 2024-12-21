# Tugas-4-IF4073-Vehicle-Recognition

Aplikasi ini adalah sebuah sistem pengenalan kendaraam dengan
menggunakan teknik-teknik di dalam pengolahan citra. Jenis kendaraan yang dijadikan objek
pengenalan adalah mobil (car), bus, dan truk (silakan tambah jenis kendaraan lainnya sebagai
bonus). Mobil adalah jenis kendaraan seperti sedan, city car, kendaraan seperti Avanza, dsb.
Terdapat dua buah program pengenalan kendaraan dengan menggunakan Matlab atau Python.
Program pertama menggunakan metode konvensional, mengggunakan teknik-teknik di dalam
pengolahan citra (edge detection, segmentation, dll). Program kedua sebagai pembanding
menggunakan deep learning dengan CNN.
Pembuatan aplikasi ini bertujuan untuk memenuhi Tugas 4 Mata Kuliah Pemrosesan Citra Digital tahun pelajaran 2024/2025

## Requirement
- Terinstall [python](https://www.python.org/downloads/) versi 3 (disarankan memakai versi terbaru)
- Terinstall tkinter python. Install dengan `pip install tk`
- Terinstall numpy python. Install dengan `pip install numpy`
- Terinstall Matlab

### Download weights
Download YOLOv4 weights from AlexeyAB/darknet repository 

https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

Using pip package manager install tensorflow and tf2-yolov4 from the command line:

        pip install tensorflow
        pip install tf2-yolov4

The tf2-yolov4 package includes convert-darknet-weights command which allows to convert Darknet weights to TensorFlow weights:

        convert-darknet-weights yolov4.weights -o yolov4.h5

## Compile and Running
### CNN
1. Buka folder TUGAS-4-IF4073-VEHICLE-RECOGNITION
2. Jalankan pada terminal `python main.py`
3. Aplikasi siap digunakan
### Non-CNN
1. Jalankan aplikasi `vehiclerecognitionapp.mlapp`

## Authors
- Faris Fadhilah - 13518026 - Teknik Informatika 2018, Institut Teknologi Bandung, Bandung
- Akhmad Setiawan - 13521164 - Teknik Informatika 2021, Institut Teknologi Bandung, Bandung
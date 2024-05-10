from setuptools import setup, find_packages

setup(
    name='FaceInference',
    version='0.1',
    author='Tomáš Kalabis',
    author_email='kalabis.tom@gmail.com',
    description='A custom inference class for face detection, image spectrum translation, and facial expression recognition.',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['FaceInference/images/*.png', 'FaceInference/centerface.onnx'],
        'FaceInference': ['centerface.onnx'],
    },
    install_requires=[
        'tensorflow==2.15.1',
        'tf-keras==2.15.1',
        'deepface==0.0.79',
        'numpy',
        'opencv-python==4.8.0.76',
        'onnx==1.15.0',
        'onnxruntime==1.15.1',
        'Pillow~=9.4.0',
        'torchvision==0.15.2',
        'matplotlib~=3.7.1',
        'ipython',
        'tqdm==4.65.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)



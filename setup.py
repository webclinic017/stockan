from setuptools import setup,find_packages

setup(
    name='stockan',
    version='0.0.1',    
    description='Stock Analysis',
    url='https://github.com/beyondbond/stockan',
    author='Ted Hong',
    author_email='ted@beyondbond.com',
    license='BSD 2-clause',
    install_requires=['pandas','numpy'],
    packages = find_packages(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Investment/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
    ],
)

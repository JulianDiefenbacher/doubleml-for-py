from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

PROJECT_URLS = {
    'Documentation': 'https://docs.doubleml.org',
    'Source Code': 'https://github.com/DoubleML/doubleml-for-py',
    'Bug Tracker': 'https://github.com/DoubleML/doubleml-for-py/issues',
}

setup(
    name='DoubleML',
    version='0.8.1',
    author='Bach, P., Chernozhukov, V., Kurz, M. S., and Spindler, M.',
    maintainer='Malte S. Kurz',
    maintainer_email='malte.simon.kurz@uni-hamburg.de',
    description='Double Machine Learning in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://docs.doubleml.org',
    project_urls=PROJECT_URLS,
    packages=find_packages(),
    install_requires=[
        'joblib',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'statsmodels',
        'plotly',
    ],
    python_requires=">=3.9",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)

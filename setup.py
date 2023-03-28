from setuptools import setup, find_packages

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setup(
    name = 'rettool',
    version = '0.0.1',
    license = 'MIT',
    description = 'Tool to help in running experiments and saving logs',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = 'https://github.com/nifey/ret',
    project_urls={
        "Bug Tracker": "https://github.com/nifey/ret/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Development Status :: 3 - Alpha",
    ],
    py_modules = ['ret'],
    packages = find_packages(),
    package_data={'ret': ['templates/*']},
    install_requires = [requirements],
    entry_points = '''
        [console_scripts]
        ret=ret.cli:cli
    '''
)

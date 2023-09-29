# CDIO Thermal Cameras Documentation

These pages contains a documentation for the program developed in the project ''Characterizing Thermal Cameras'' in the CDIO course at Linköpings University produced at the end of 2022.
The customer for this research project is Termisk Systemteknik located in Linköping, Sweden, and the developers of the project are Maja Boström, Johanna Carlson, William Torberntsson and Kim Nguyên Sundström. 
For further reading and explaination of how measurements should be performed in order to use the program properly, please see the Project Report.

## Contents



1. [Setting up Environment](how-to-install-program.md)
2. [Running the Program](how-to-run-program.md)
3. [Exampel of Running the Program](example.md)
4. [Distorsion](tutorial_dist.md)
5. [MTF - Modulation Transfer Function](tutorial_mtf.md)
6. [NEDT - Noise Equivalent Differential Temperature](tutorial_nedt.md)
7. [Standard Deviation of Temperature](tutorial_temp.md)



## Layout

    mkdocs.yml    # The configuration file.
    docs/
        tutorials/
            dist.md
            mtf.md
            nedt.md
            temp.md
        example.md
        how-to-install-program.md # Documentation of how to install the program
        how-to-run-program.md # Documentation of how to run the program
        index.md  # The documentation homepage.
    scripts/
        distortion.py # Calculates distortion
        main.py # Main programming handling arguments
        mtf.py # Calculates MTF
        nedt.py # Calculates NEDT
        temperature.py # Calculates std of temperature


## Overview

::: scripts
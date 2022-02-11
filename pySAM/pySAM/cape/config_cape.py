import numpy as np

# CONSTANT FOR CALCULATING SATURATION MIXING RATIO

## Polynomial Fits to Saturation Vapor Pressure, Piotr J. Flatau et al 1992,
###Tab 3: Coefficients of the sixth-order polynomial fits to SVP for the temperature range -50°C to 50°C


AW0 = 6.11239921
AW1 = 443987641
AW2 = 0.142986287 * 10 ** -1
AW3 = 0.264847430 * 10 ** -3
AW4 = 0.302950461 * 10 ** -5
AW5 = 0.206739458 * 10 ** -7
AW6 = 0.640689451 * 10 ** -10
AW7 = -0.952447341 * 10 ** -13
AW8 = -0.976195544 * 10 ** -15

AW = [AW0, AW1, AW2, AW3, AW4, AW5, AW6, AW7, AW8]

### Same for ice but for the temperature range -50°C to 0°C


AI0 = 6.11147274
AI1 = 503160820
AI2 = 0.188439774 * 10 ** -1
AI3 = 0.420895665 * 10 ** -3
AI4 = 0.615021634 * 10 ** -5
AI5 = 0.602588177 * 10 ** -7
AI6 = 0.385852041 * 10 ** -9
AI7 = 0.146898966 * 10 ** -11
AI8 = 0.252751365 * 10 ** -14

AI = [AI0, AI1, AI2, AI3, AI4, AI5, AI6, AI7, AI8]

## REFERENCE TEMPERATURE

T0_KELVIN = 273.16  # K

# mixing ratio
MIXING_RATIO_AIR_WATER_VAPOR = 0.622  # (%)


# ARDEN BUCK FORMULA EQUATION FOR THE SATURATION VAPOR PRESSURE FOR MOIST AIR
# link to wikipedia : https://en.wikipedia.org/wiki/Arden_Buck_equation
# Ps(T)=6.1121exp((18.678-T/234.5)(T/(257.14+T))) over liquid water T>0°C
# Ps(T)=6.1115exp((23.036-T/333.7)(T/(279.82+T))) over ice T<0°C

AW = 6.1121
BW = 18.678
CW = 257.14
DW = 234.5


AI = 6.1115
BI = 23.036
CI = 279.82
DI = 333.7

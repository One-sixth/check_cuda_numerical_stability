# check_cuda_numerical_stability
 A script uses the principle of IREVNET to detect whether your CUDA computing card is normal.  


# How to check
This tool works by the reversibility principle of IREVNET.  
In the current implementation, the result is calculated by the IREV module first, and then the result is inverted by the IREV module to restore the input.  
By detecting the difference between the maximum value of the input and the reconstructed input, it can be judged whether there is a numerical error in the CUDA card.  

In normal graphics cards, the numerical error will continue to be less than 1e-5. On abnormal graphics cards, the numerical error will occasionally exceed 1e-3.  
In my local test, when using a normal graphics card, it can pass the test for 2 hours without any errors, and I haven't tried for a longer time. When using abnormal graphics cards, the test often reports errors within 5-25 minutes.  


# Dependent
```
python3
pytorch >= 1.1
argparse
```

# Command
```
python .\_check_cuda_numerical_stability.py -h
usage: _check_cuda_numerical_stability.py [-h] [-i I] [-t T] [-bs BS]

Used to detect CUDA numerical stability problems.

optional arguments:
  -h, --help  show this help message and exit
  -i I        card id. Which cuda card do you want to test. default: 0
  -t T        minute. Test duration. When the setting is less than or equal to 0, it will not stop automatically.
              defaule: 30
  -bs BS      Test batch size when testing. defaule: 20
```

# How to use

```
python _check_cuda_numerical_stability.py
```
This command will start a test immediately. By default, card 0 will be detected for 30 minutes.  
If there is no error within 30 minutes, "Test passed" will be output, which means your card may be no problem.  
Otherwise, it will be interrupted prematurely and the "Test failure" will be output, which means that your CUDA card may have some problems or the slot is not securely inserted.  

```
python _check_cuda_numerical_stability.py -i 1 -t 60
```
This command specifies that the card 1 will be tested, and the duration is 60 minutes.  

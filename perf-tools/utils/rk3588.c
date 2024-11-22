#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    FILE* ptr;
    signed char ch;
    int end = atoi(argv[1]);

    while(end) {
        printf("AN-rk3588,");
        // Opening file in reading mode
        ptr = fopen("/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq", "r");
        // Printing what is written in file
        // character by character using loop.
        while (1) {
            ch = fgetc(ptr);
            // Checking if character is not EOF.
            // If it is EOF stop reading.
            if (ch == '\n') break;
            printf("%c", ch);
        }
        // Closing the file
        fclose(ptr);
        printf(",");
        ptr = fopen("/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq", "r");
        while (1) {
            ch = fgetc(ptr);
            if (ch == '\n') break;
            printf("%c", ch);
        }
        fclose(ptr);
        printf(",");
        ptr = fopen("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq", "r");
        while (1) {
            ch = fgetc(ptr);
            if (ch == '\n') break;
            printf("%c", ch);
        }
        fclose(ptr);
        printf(",");
        ptr = fopen("/sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/cur_freq", "r");
        while (1) {
            ch = fgetc(ptr);
            if (ch == '\n') break;
            printf("%c", ch);
        }
        fclose(ptr);
        printf(",");
        ptr = fopen("/sys/devices/platform/fb000000.gpu/utilisation", "r");
        while (1) {
            ch = fgetc(ptr);
            if (ch == '\n') break;
            printf("%c", ch);
        }
        fclose(ptr);
        printf("\n");

        usleep(1*1000*1000);
        end--;
    }
    return 0;
}
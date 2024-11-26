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
        printf("devboard-s905d3,");
        ptr = fopen("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq", "r");
        while (1) {
            ch = fgetc(ptr);
            if (ch == '\n') break;
            printf("%c", ch);
        }
        fclose(ptr);
        printf(",");
        ptr = fopen("/sys/class/mpgpu/cur_freq", "r");
        while (1) {
            ch = fgetc(ptr);
            if (ch == '\n') break;
            printf("%c", ch);
        }
        fclose(ptr);
        printf(",");
        ptr = fopen("/sys/class/mpgpu/utilization", "r");
        while (1) {
            ch = fgetc(ptr);
            if (ch == '\n') break;
            printf("%c", ch);
        }
        fclose(ptr);
        printf(",");
        ptr = fopen("/sys/class/mpgpu/util_cl", "r");
        while (1) {
            ch = fgetc(ptr);
            if (ch == '\n') break;
            printf("%c", ch);
        }
        fclose(ptr);
        printf(",");
        ptr = fopen("/sys/class/mpgpu/util_gl", "r");
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
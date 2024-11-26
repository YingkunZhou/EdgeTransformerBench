# the script for merge power-z log output and the testsuite performance log
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python perf_power_merged.py <perf_file_path> <power_file_path>")
        sys.exit(1)
    perf_file_path = sys.argv[1]
    power_file_path = sys.argv[2]
    perf_logs = open(perf_file_path).readlines()
    power_logs = open(power_file_path).readlines()
    output_csv_file = perf_file_path[:-4] + '-merged' + '.log'
    prev_pow = 0
    i = 0
    start = True
    merged_output = ''
    while start:
        if '(index: ' in perf_logs[i] or '[(' == perf_logs[i][:2]:
            start = False
        merged_output += perf_logs[i]
        i+=1
    merged_output += 'power-z measurement log is merged:\n'

    damping = 1
    for pl in power_logs[1:]:
        pls = pl.strip().split(',')
        if pls:
            cur_pow = float(pls[-1])
            start = (prev_pow - cur_pow) > 3 and damping == 0
            while start and i < len(perf_logs):
                damping = 20
                merged_output += perf_logs[i]
                if '(index: ' in perf_logs[i] or '[(' == perf_logs[i][:2]:
                    start = False
                i += 1
            merged_output += pl
            prev_pow = cur_pow
            if damping > 0: damping -= 1

    open(output_csv_file, "w").write(merged_output)

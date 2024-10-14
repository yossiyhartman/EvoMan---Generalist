class Logger:
    def __init__(self, headers, print=True) -> None:

        # All headers
        self.headers = headers
        self.logs = {h: [] for h in headers}

        # header to print
        self.lengths = [len(header) for header in headers]

        # Turn off if you don't want a print
        self.print = print

    def print_headers(self):
        if self.print:
            header_row = " | ".join(str(header).ljust(length) for length, header in zip(self.lengths, self.headers))
            print("\n" + header_row)
            print("-" * len(header_row))

    def print_log(self, log):
        print(" | ".join(str(value).ljust(length) for length, value in zip(self.lengths, log)))

    def save_log(self, log):
        # print the logs
        if self.print:
            self.print_log(log.values())

        for k, v in log.items():
            self.logs[k].append(v)

    def clean_logs(self):
        self.logs = {h: [] for h in self.headers}

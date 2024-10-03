class Logger:
    def __init__(self, headers) -> None:
        self.headers = headers
        self.lengths = [len(header) for header in headers]

    def print_headers(self):
        header_row = " | ".join(str(header).ljust(length) for length, header in zip(self.lengths, self.headers))
        print("\n" + header_row)
        print("-" * len(header_row))

    def print_log(self, log):
        print(" | ".join(str(value).ljust(length) for length, value in zip(self.lengths, log)))

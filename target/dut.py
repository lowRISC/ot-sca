import serial
from serial.tools.list_ports import comports


class DUT:
    def __init__(self):
        baudrate = 115200

        def find_DUT_port():
            for port in comports():
                if "UART2" in port.description:
                    return port.device
            print("DUT not found!")
            return None

        self.com_interface = self._init_communication(find_DUT_port(), baudrate)

    def _init_communication(self, port, baudrate):
        """Open the communication interface.

        Configure OpenTitan on CW FPGA or the discrete chip.
        """
        com_interface = None
        com_interface = serial.Serial(port)
        com_interface.baudrate = baudrate
        com_interface.timeout = 1
        return com_interface

    def write(self, data):
        """Write data to the target."""
        self.com_interface.write(data)

    def readline(self):
        """read line."""
        return self.com_interface.readline()

    def print_all(self):
        while True:
            read_line = str(self.readline().decode().strip())
            if len(read_line) > 0:
                print(read_line, flush=True)
            else:
                break

    def dump_all(self):
        while True:
            read_line = str(self.readline())
            if len(read_line) <= 5:
                break

    def check_crash_or_read_reponse(self, max_tries=50):
        """
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            The JSON response of OpenTitan.
        """
        it = 0
        while it != max_tries:
            try:
                read_line = str(self.readline())
                if "FAULT" in read_line:
                    return read_line, False
                if "RESP_OK" in read_line:
                    return read_line.split("RESP_OK:")[1].split(" CRC:")[0], True
                it += 1
            except UnicodeDecodeError:
                it += 1
                continue
        return "", False

    def read_response(self):
        """
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            The JSON response of OpenTitan.
        """
        it = 0
        while it < 20:
            try:
                read_line = str(self.readline().decode().strip())
            except UnicodeDecodeError:
                break
            if len(read_line) > 0:
                if "RESP_OK" in read_line:
                    return read_line.split("RESP_OK:")[1].split(" CRC:")[0]
            else:
                break
            it += 1
        return ""

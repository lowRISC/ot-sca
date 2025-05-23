import serial
from subprocess import Popen
from serial.tools.list_ports import comports

class DUT:
    def __init__(self):
        baudrate = 115200
        
        def find_DUT_port():
            for port in comports():
                if 'UART2' in port.description:
                    return port.device
            print("DUT not found!")
            return None
        
        self.com_interface = self._init_communication(find_DUT_port(), baudrate)
        
    def _init_communication(self, port, baudrate):
        """ Open the communication interface.

        Configure OpenTitan on CW FPGA or the discrete chip.
        """
        com_interface = None
        com_interface = serial.Serial(port)
        com_interface.baudrate = baudrate
        com_interface.timeout = 1
        return com_interface

    def write(self, data):
        """Write data to the target. """
        self.com_interface.write(data)

    def readline(self):
        """read line. """
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
            read_line = str(self.readline().decode().strip())
            if len(read_line) <= 0:
                break
    
    def read_response(self, max_tries = 1):
        """ Read response from AES SCA framework.
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            The JSON response of OpenTitan.
        """
        it = 0
        while it != max_tries:
            read_line = str(self.readline())
            if "RESP_OK" in read_line:
                return read_line.split("RESP_OK:")[1].split(" CRC:")[0]
            it += 1
        return ""


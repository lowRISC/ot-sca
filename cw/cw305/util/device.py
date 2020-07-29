r"""CW305 utility functions. Used to configure FPGA with OpenTitan design."""
import subprocess
import time

import chipwhisperer as cw


SPIFLASH=r'bin/linux/spiflash'


class OpenTitan(object):
  def __init__(self, bitstream, fw_image, pll_frequency, baudrate):
      self.bitstream = bitstream
      self.fw_image = fw_image
      self.pll_frequency = pll_frequency
      self.initialized = False
      self.fpga = None
      self.scope = None
      self.target = None
      self.baudrate = baudrate

  def initialize(self):
    """Initializes FPGA target."""
    self.initialize_fpga()
    self.initialize_scope()
    self.initialize_target()

  def initialize_fpga(self):
    """Initializes FPGA bitstream and sets PLL frequency."""
    print('Connecting and loading FPGA')
    self.fpga = cw.capture.targets.CW305()
    self.fpga.con(bsfile=self.bitstream, force=True)
    self.fpga.vccint_set(1.0)

    print('Initializing PLL1')
    self.fpga.pll.pll_enable_set(True)
    self.fpga.pll.pll_outenable_set(False, 0)
    self.fpga.pll.pll_outenable_set(True, 1)
    self.fpga.pll.pll_outenable_set(False, 2)
    self.fpga.pll.pll_outfreq_set(self.pll_frequency, 1)

    # 1ms is plenty of idling time
    self.fpga.clkusbautooff = True
    self.fpga.clksleeptime = 1

  def initialize_scope(self):
    """Initializes chipwhisperer scope."""
    self.scope = cw.scope()
    self.scope.gain.db = 25
    self.scope.adc.samples =  200
    self.scope.adc.offset = 0
    self.scope.adc.basic_mode = "rising_edge"
    self.scope.clock.clkgen_freq = 18425000
    self.scope.clock.adc_src = "extclk_x4"
    self.scope.trigger.triggers = "tio4"
    self.scope.io.tio1 = "serial_tx"
    self.scope.io.tio2 = "serial_rx"
    self.scope.io.hs2 = "disabled"

    # TODO: Need to update error handling.
    self.scope.clock.reset_adc()
    time.sleep(0.5)
    assert (self.scope.clock.adc_locked), "ADC failed to lock"

  def load_fw(self):
    """Loads firmware image."""
    # TODO: Make settings configurable.
    command = [SPIFLASH, '--dev-id=0403:6014', '--dev-sn=FT2U2SK1',
           '--input=' + self.fw_image]
    subprocess.check_call(command)

  def initialize_target(self):
    """Loads firmware image and initializes test target."""
    self.load_fw()
    time.sleep(0.5)
    self.target = cw.target(self.scope)
    self.target.output_len = 16
    self.target.baud = self.baudrate
    self.target.flush()

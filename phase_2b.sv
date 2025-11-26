module PHASE_2b #(
  parameter BITSTREAM = 64
)(
  input  logic [1 : 0]          k,
  input  logic [BITSTREAM-1:0]  in_bits,
  output logic [BITSTREAM-1:0]  out_bits
);

  logic [BITSTREAM-1:0] s1, s2;
  assign s1 = k[0] ? { in_bits[BITSTREAM-2:0],      in_bits[BITSTREAM-1] } : in_bits;
  assign s2 = k[1] ? { s1[BITSTREAM-3:0],           s1[BITSTREAM-1:BITSTREAM-2] } : s1;
  assign out_bits = s2;
endmodule

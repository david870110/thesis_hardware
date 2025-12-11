module PHASE_3b #(
  parameter BITSTREAM = 64
)(
  input  logic [2 : 0]          k,
  input  logic [BITSTREAM-1:0]  in_bits,
  output logic [BITSTREAM-1:0]  out_bits
);
  logic [BITSTREAM-1:0] s1, s2, s3;
  assign s1 = k[0] ? { in_bits[BITSTREAM-2:0],      in_bits[BITSTREAM-1] } : in_bits;
  assign s2 = k[1] ? { s1[BITSTREAM-3:0],           s1[BITSTREAM-1:BITSTREAM-2] } : s1;
  assign s3 = k[2] ? { s2[BITSTREAM-5:0],           s2[BITSTREAM-1:BITSTREAM-4] } : s2;
  assign out_bits = s3;
endmodule
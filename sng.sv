// 42 GE
module SNG #(
    parameter BITSTREAM = 64,
    parameter BASE      = 61,
    parameter STRIDE    = 17,
    parameter QUANT     = 8
)(
    input   [QUANT - 1: 0]      w_data,
    input                       clk,
    input                       rst_n,
    input                       w_valid,
    input                       r_ready,
    input                       wlast,
    output  [BITSTREAM-1 : 0]   r_bitstream
);
    logic [$clog2(BITSTREAM) : 0 ] quota_num;
    logic [BITSTREAM - 1 : 0] weyl_bit;
    logic [1:0] k;
    logic [BITSTREAM - 1 : 0] phase_bit;

    QUOTA #(
        .BITSTREAM  (BITSTREAM),
        .QUANT      (QUANT)
    ) quota_dut (
        .data   (w_data),
        .quota  (quota_num)
    );

    WEYL #(
        .BITSTREAM (BITSTREAM),
        .BASE      (BASE),
        .STRIDE    (STRIDE)
    ) weyl_dut (
        .quota_num (quota_num),
        .weyl_out  (weyl_bit)
    );

    always_ff @(posedge clk or negedge rst_n)
        if(!rst_n) k <= 0;
        else if(r_ready & w_valid) k <= k+1;

    PHASE_2b #(
        .BITSTREAM  (BITSTREAM)
    ) dut (
        .k          (k),
        .in_bits    (weyl_bit),
        .out_bits   (phase_bit)
    );

endmodule
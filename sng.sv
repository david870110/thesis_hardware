// 42 GE
module SNG #(
    parameter BITSTREAM = 64,
    parameter BASE      = 61,
    parameter STRIDE    = 17,
    parameter QUANT     = 8
)(
    input   [QUANT - 1: 0]         iData,
    output  [BITSTREAM-1 : 0]      oBitstream
);

    logic [$clog2(BITSTREAM) : 0 ] quota_num;



    QUOTA #(
        .BITSTREAM  (BITSTREAM),
        .QUANT      (QUANT)
    ) quota_dut (
        .data   (iData),
        .quota  (quota_num)
    );


    WEYL #(
        .BITSTREAM (BITSTREAM),
        .BASE      (BASE),
        .STRIDE    (STRIDE)
    ) weyl_dut (
        .quota_num (quota_num),
        .weyl_out  (oBitstream)
    );
    


endmodule
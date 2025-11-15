// 42 GE
module QUOTA #(
    parameter BITSTREAM = 64,
    parameter QUANT     = 8
)(
    input   [QUANT - 1: 0]                data,
    output  [$clog2(BITSTREAM) : 0]      quota
);
    localparam QUANT_DIV_T = QUANT - $clog2(BITSTREAM);

    initial begin
        if ((1 << $clog2(BITSTREAM) ) != BITSTREAM)  $error("Bitstream must be 2^n ");
        if (QUANT_DIV_T < 0)                         $error("Bitstream log 2 < quant log 2");
    end


    logic [QUANT : 0 ] bias_data,round_data;

    // bias data
    assign bias_data = data + (9'd1 << (QUANT-1));

    // rounding
    assign round_data = bias_data + (9'd1 << (QUANT_DIV_T - 1));

    // quota
    assign quota = round_data >> QUANT_DIV_T;
    
endmodule
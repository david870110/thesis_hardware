// 240GE
module WEYL #(
    parameter BITSTREAM = 64,
    parameter BASE      = 61,
    parameter STRIDE    = 17
)(
    input  [$clog2(BITSTREAM)-1 : 0]    quota_num,
    output [BITSTREAM-1 : 0]            weyl_out
);
    wire [BITSTREAM-1 : 0 ] one_hot,quota_mask;

    // one-hot
    assign one_hot = ({{(BITSTREAM-1){1'b0}},1'b1}) << quota_num;

    // thermometer
    assign quota_mask[0] = 1'b0;
    generate
        genvar k;
        for (k = 1; k < BITSTREAM; k = k + 1) begin : GEN_THERM
            assign quota_mask[k] = quota_mask[k-1] | one_hot[k-1];
        end
    endgenerate
    
    // stride phase and connect wire
    generate
        genvar i;
        for (i = 0; i < BITSTREAM; i++) begin : GEN_TABLE
            localparam int IDX = (BASE + (i*STRIDE)) % BITSTREAM; 
            assign weyl_out[IDX] = quota_mask[i];
        end
    endgenerate
endmodule
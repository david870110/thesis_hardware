// 240GE
module WEYL #(
    parameter BITSTREAM = 64,
    parameter BASE      = 61,
    parameter STRIDE    = 17
)(
    input  unsigned [$clog2(BITSTREAM) : 0]      quota_num,
    output [BITSTREAM-1 : 0]            weyl_out
);
    logic [BITSTREAM-1 : 0 ] one_hot,quota_mask;

    // one-hot 
    assign one_hot = ({{(BITSTREAM-1){1'b0}},1'b1}) << quota_num[$clog2(BITSTREAM) - 1:0];

    // ★ 這裡從 0 改成 full64
    assign quota_mask[BITSTREAM-1] = quota_num[$clog2(BITSTREAM)];

    // 後面不變
    genvar k;
    generate
    for (k = (BITSTREAM-1); k >= 0; k = k - 1) begin
        assign quota_mask[k-1] = quota_mask[k] | one_hot[k];
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
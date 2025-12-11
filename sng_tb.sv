`timescale 1ns/1ps


module SNG_TB;
    // ---------------------------------------------------------------------------
    // Parameters for DUT instantiation (override with +define+TB_*)
    // ---------------------------------------------------------------------------
    localparam int TB_BITSTREAM  = 64;
    localparam int TB_BASE       = 2;
    localparam int TB_STRIDE     = 17;
    localparam int TB_QUANT      = 8;

    // ---------------------------------------------------------------------------
    // DUT I/O
    // ---------------------------------------------------------------------------
    logic [TB_QUANT-1:0]        iData;
    logic [TB_BITSTREAM-1:0]    oBitstream;

    // ---------------------------------------------------------------------------
    // DUT instantiation
    // ---------------------------------------------------------------------------
    SNG #(
        .BITSTREAM  (TB_BITSTREAM),
        .BASE       (TB_BASE),
        .STRIDE     (TB_STRIDE),
        .QUANT      (TB_QUANT)
    ) dut (
        .iData   (iData),
        .oBitstream  (oBitstream)
    );

    // ---------------------------------------------------------------------------
    // Reference model (mirrors the RTL intent)
    // u = q + 128
    // s = (u * T + (1<< (QUANT-1) ) ) >> QUANT
    // return s
    // ---------------------------------------------------------------------------
    logic clk, rst_n;
    always #50 clk = ~clk;

    function automatic logic [TB_BITSTREAM - 1:0] ref_sng (int q);
        logic [TB_BITSTREAM - 1 : 0] tb_bitstream;
        int idx , u , s;
        begin
            //quota 
            u = q + (1 << (TB_QUANT-1));
            s = (u * TB_BITSTREAM + (1<< (TB_QUANT-1) ) ) >> TB_QUANT;
            
            //weyl
            tb_bitstream = '0;
            idx = TB_BASE;
            for (int i = 0; i < s; i++) begin
                tb_bitstream[idx] = 1'b1;
                idx = (TB_STRIDE + idx) % TB_BITSTREAM ;
            end 
            return tb_bitstream;
        end
    endfunction





    integer errors = 0;
    localparam int DATA_RANGE_NEG = -(1 << (TB_QUANT-1));         // -128
    localparam int DATA_RANGE_POS =  (1 << (TB_QUANT-1)) - 1;     //  127

    // Optional VCD/FSDB dump
    initial begin
    `ifdef DUMPVCD
        $dumpfile("sng_tb.vcd");
        $dumpvars(0, SNG_TB);
    `endif
    end

    // Main test sequence: sweep all q, then random samples
    initial begin
        $display("\n=== SNG_TB TB start ===");
        $display("Params: BITSTREAM=%0d BASE=%0d STRIDE=%0d QUANT=%0d", TB_BITSTREAM, TB_BASE, TB_STRIDE, TB_QUANT);


        // Deterministic sweep
        for (int q = DATA_RANGE_NEG; q <= DATA_RANGE_POS; q++)begin
            logic [TB_BITSTREAM-1:0] golden;

            iData = q;
            #1; // settle (purely combinational)
            golden = ref_sng(q);

            if (oBitstream !== golden) begin
                errors++;
                $error("Mismatch at q=%0d: design=%0d tb=%0d", q, oBitstream, golden);
            end
        end

        // A few randoms (in-range)
        // for (int t = 0; t < 1000; t++) begin
        //     logic [$clog2(TB_BITSTREAM)-1:0] golden;
        //     int q = $urandom_range(DATA_RANGE_NEG, DATA_RANGE_POS);
        //     data = q;
        //     #1; // settle (purely combinational)
        //     golden = ref_quota(q);

        //     if (quota_out !== golden) begin
        //         errors++;
        //         $error("Mismatch at q=%0d: design=%0d tb=%0d", q, quota_out, golden);
        //     end
        // end

        if (errors == 0)begin
            $display("\n[PASS] All checks passed.\n");
            $finish;
        end
        else begin
            $display("\n[FAIL] %0d mismatches detected.\n", errors);
            $fatal(1);
        end
    end
endmodule

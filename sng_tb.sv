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
    logic                       clk,rst_n;
    logic                       w_valid;
    logic                       r_ready;
    logic                       wlast;
    logic [TB_QUANT-1:0]        w_data;
    logic [TB_BITSTREAM-1:0]    r_bitstream;

    // ---------------------------------------------------------------------------
    // DUT instantiation
    // ---------------------------------------------------------------------------
    SNG #(
        .BITSTREAM  (TB_BITSTREAM),
        .BASE       (TB_BASE),
        .STRIDE     (TB_STRIDE),
        .QUANT      (TB_QUANT)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .w_data         (w_data),
        .w_valid        (w_valid),
        .r_ready        (r_ready),
        .wlast          (wlast),
        .r_bitstream    (r_bitstream)
    );

    always #50 clk = ~clk;

    // ---------------------------------------------------------------------------
    // generate data
    // ---------------------------------------------------------------------------
    logic [TB_QUANT-1 : 0] test_data;
    logic test_last,test_valid,test_ready;
    task automatic generate_data;
        input[TB_QUANT-1 : 0] data;
        input last,valid,ready;
        begin
            test_data = data;
            test_last = last;
            test_valid = valid;
            test_ready = ready;
            @(posedge clk);
        end
    endtask

    task random_data_generate;
        input int unsigned total_loop;
        begin
            for(int i = 0 ; i < total_loop ; i = i+1)
            begin
                int rand_val;
                rand_val = $urandom_range(128, -127);
                generate_data(rand_val,$urandom_range(1, 0),1,1);
            end
        end
    endtask

    // ---------------------------------------------------------------------------
    // assign input : test data assign to design
    // ---------------------------------------------------------------------------
    assign w_data   = test_data;
    assign w_valid  = test_valid;
    assign r_ready  = test_ready;
    assign wlast    = test_last;

    // ---------------------------------------------------------------------------
    // Reference model
    // input : data / last / w_valid / r_ready
    // combination for weyl + k
    // ---------------------------------------------------------------------------
    logic [TB_BITSTREAM - 1 : 0] exp_data;
    integer k_phase = 0 ;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            k_phase  <= 0;
        else if (test_valid && test_ready) begin  
            if (test_last)      k_phase <= 0;        
            else                k_phase <= (k_phase + 1) & 3;
        end
    end

    function automatic logic [TB_BITSTREAM - 1:0] ref_sng (int tb_data,bit tb_last,bit tb_valid,bit tb_ready);
        logic [TB_BITSTREAM - 1 : 0] tb_bitstream;
        int idx , u , s;
        begin
            if(tb_valid & tb_ready) begin
                //quota 
                u = tb_data + (1 << (TB_QUANT-1));
                s = (u * TB_BITSTREAM + (1 << (TB_QUANT-1) ) ) >> TB_QUANT;
                
                //weyl
                tb_bitstream = '0;
                idx = TB_BASE + k_phase;
                for (int i = 0; i < s; i++) begin
                    tb_bitstream[idx] = 1'b1;
                    idx = (TB_STRIDE + idx) % TB_BITSTREAM ;
                end 
            end
            else
                tb_bitstream = 'bz;
            return tb_bitstream;
        end
    endfunction
    
    assign exp_data = ref_sng($signed(test_data),test_last,test_valid,test_ready);
    // ---------------------------------------------------------------------------
    // tool : auto check
    // ---------------------------------------------------------------------------
    localparam DATA_WIDTH = TB_BITSTREAM;
    logic [DATA_WIDTH-1 : 0] design_data;
    assign design_data = r_bitstream;

    int errors = 0;
    task automatic auto_check;
        begin
            if(exp_data !== design_data)
            begin
                errors++;
                $display("    ERROR : auto check compare result is failed, \n input_data : %d \n exp_data : %h , \n design_data : %h." ,test_data , exp_data, design_data);
                $stop;
            end
        end
    endtask

    // ---------------------------------------------------------------------------
    // testing task
    // ---------------------------------------------------------------------------
    task automatic all_range_check;
        for(int i = 0 ; i < 3000 ; i = i+1)
        begin
            random_data_generate(1);
            auto_check();
        end
    endtask
    task automatic reset;
        begin
            clk     = 0;
            rst_n   = 1;
            @(posedge clk)
            rst_n   = 0;
            @(posedge clk)
            rst_n   = 1;
        end
    endtask


    // ---------------------------------------------------------------------------
    // main
    // ---------------------------------------------------------------------------

    initial begin
    `ifdef DUMPVCD
        $dumpfile("sng_tb.vcd");
        $dumpvars(0, SNG_TB);
    `endif
    end

    initial begin
        $display("\n=== SNG_TB TB start ===");
        $display("Params: BITSTREAM=%0d BASE=%0d STRIDE=%0d QUANT=%0d", TB_BITSTREAM, TB_BASE, TB_STRIDE, TB_QUANT);
        reset();
        all_range_check();

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

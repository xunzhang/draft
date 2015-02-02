Cosine similarity is a way to measure the relation degree between two objects.
This simple algorithm works very well in industry of recommendation systems. In
real applications, since dataset is large, you always need to implement it using
a distributed processing system such as hadoop and spark. You can also use
low-level tool such as MPI to parallelize the algorithm yourself. At douban, we
can use paracel. Paracel is a distributed optimization framework which provides
easy to use communication interfaces with parameter servers. This post will
introduce the distributed implementation of cosine similarity algorithm using
paracel.

Basic concept. Cosine similarity is really a simple idea that use the cosine
value of the angle between two vectors to represent the similarity of them. For
example, a user vector can be constructed using activity values(rating,
                                                                clicking, etc.)
  in item space of the website. Also, a vector can be constructed by using some
  predefined or underlying(ML techniques) dimensions. After calculating out the
  similarities, we can do clustering, recommendation and a lot of other valuable
  things.

  Anatomy. Suppose the input file consist of a collection of triples: {(uid,
                                                                        iid,
                                                                        value)}.
  The data now construct a sparse matrix A with each row represents a user
  vector and each column represents a item vector. We want to calculate the
  cosine similarities of every two rows. Firstly, we need to normalize the
  vector to make sure that value in each dimension at same scale. Secondly, we
  can do kinds of transformation on the vector: Cosine(1/norm2(v(u,i))), Pearson
  Correlation((x(u,i)-x_bar(i))/(norm2(x(u,i)-x_bar(i)))), Adjusted Cosine
  Similarity((x(u,i)-x_bar(u))/(norm2(x(u,i)-x_bar(u))))...Thirdly, we will just
  do matrix multiply: A * A'(or A * B', B is a "sub matrix" of A).

  Complexity. rou is the sparsity of the A, so the complexity of this algorithm
  is depending on the matrix multiply stage: O(m*m*n*rou) or O(n*n*m*rou), here
  rou may be 0.1%, website m may be hundreds of millions users, n may be tens of
  millions items. It is obviously cpu intensive so we are always need to do
  parallelization. The intuitive strategy to parallize a matrix multiply is
  partition A by rows. In this case, you must broadcast the data of each
  node/process to every other node/process. Another strategy is partiton A by
  columns. In this case, every node/process do the matrix multiply locally and
  need a reduce process to get final result. Before coding, let's do some
  estimation first. In both methods, the compute time is linear scalable. The
  communication cost term in planA will be O(m/p*n*rou*p*p) which in planB will
  be O(m*m*p). But p processes do reduction in planB at the same time so the
  term of T(process) will be O(m*m). So T(p) is monotonically decreasing(but a
                                                                         little
                                                                         large
                                                                         then
                                                                         planA
                                                                         since
                                                                         dense).
  The other curve of scalablity T(datasize) of planB is little tricky because
  the m*m matrix can not reside locally. We can reuse the memory/stack space by
  spliting the rows locally or using parameter server. Since planA is
  implemented in doubanm(internal algorithm library at douban), we will
  implement planB in this post.

  Paracel implementation.
  class cos_sim_sparse : public paracel::paralg {
   public:
    cos_sim_sparse(paracel::Comm comm, 
                   string hosts_dct_str,
                   string
                   _input, 
                   string
                   _output)
        : 
            paracel::paralg(hosts_dct_str, comm, _output),
            input(_input),
            output(_output) {}

    virtual void solve() {
      auto f_parser = paracel::gen_parser(local_parser);
      paracel_load_as_matrix(blk_A_T, row_map, input, f_parser,
                             "smap");
      normalize();

      blk_size = blk_A_T.rows() / 5000; // split rows
      locally
          if(blk_size == 0) {
            blk_size = 1;
          }
      for(int k = 0; k < blk_size; ++k) {
        int cols = blk_A_T.cols();
        int rows = blk_A_T.rows() /
            blk_size;
        if(k == blk_size - 1) {
          rows +=
              blk_A_T.rows()
              % blk_size;
        }
        Eigen::MatrixXd
            part_blk_A_T =
            Eigen::MatrixXd(blk_A_T).block(k
                                           * blk_size,
                                           * 0,
                                           * rows,
                                           * cols);
        Eigen::MatrixXd
            local_result
            =
            part_blk_A_T
            * blk_A_T.transpose();
        vector<double>
            vec_buff
            =
            paracel::mat2vec(local_result);
        paracel_bupdate("result_"
                        +
                        std::to_string(k), 
                        vec_buff, 
                        "/mfs/user/wuhong/paracel/local/lib/libcos_sim_sparse_update.so", 
                        "cos_sim_sparse_updater");
      }
      sync();
    }
   private:
    string input, output;
    Eigen::SparseMatrix<double, Eigen::RowMajor> blk_A_T;
    unordered_map<size_t, string> row_map;
    int blk_size = 0;
    vector<Eigen::MatrixXd> result;
  }; // class cos_sim_sparse

You are almost done, transformation here need to get the total row information,
    so we need to load the data by fmap and store the weights into parameter
    servers:
    void normalize() {
      Eigen::SparseMatrix<double, Eigen::RowMajor> blk_A;
      unordered_map<size_t, string> A_rm;
      auto f_parser = paracel::gen_parser(local_parser);
      paracel_load_as_matrix(blk_A, A_rm, input, f_parser,
                             "fmap");

      vector<double> wgt(blk_A.rows(), 0);
      auto lambda = [&] (int i, int j, double & v) {
        wgt[i] += v * v;
      };
      paracel::traverse_matrix(blk_A, lambda);
      for(size_t i = 0; i < wgt.size(); ++i) {
        paracel_write("wgt_" + A_rm[i],
                      1. /
                      sqrt(wgt[i]));
      }
      wgt.resize(0);
      A_rm.clear();
      blk_A.resize(0, 0);
      sync();

      auto wgt_map =
          paracel_read_special<double>("libcos_sim_sparse_update.so",
                                       "cos_sim_sparse_filter");
      auto norm_lambda
          = [&] (int
                 i,
                 int
                 j,
                 double
                 & v)
          {
            v *=
                wgt_map["wgt_"
                +
                row_map[i]];
          };
      paracel::traverse_matrix(blk_A_T,
                               norm_lambda);
      sync();
    }

The update/filter function in the above code may look like this:

std::vector<double> local_update(const std::vector<double> & a, 
                                 const
                                 std::vector<double>
                                 & b) {
  std::vector<double> r(a);
  for(size_t i = 0; i < b.size(); ++i) {
    r[i] += b[i];
  }
  return r;
}

bool filter(const std::string & key) {
  std::string s = "wgt_";
  if(paracel::startswith(key, s)) {
    return true;
  }
  return false;
}

Bottomline. We can also use graph data structure(paracel_load_as_graph
                                                 interface) to implement the
algorithm, I think it's more flexible to iterate the data. You must add sort
logic and some filter logic(simbar) to make your code work in your situation.

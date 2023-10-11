#ifndef _utiles
#define _utiles



template <class T>
void print_array(T* array, u16 rows, u16 cols)
{
    for(u16 r = 0; r < rows; r++)
    {
        for(u16 c = 0; c < cols; c++)
            std::cout<<(fpt)array[c+r*cols]<<" ";
        printf("\n");
    }
}

template <class T>
void print_named_array(const char* name, T* array, u16 rows, u16 cols)
{
    printf("%s:\n",name);
    print_array(array,rows,cols);
}

//print named variable
template <class T>
void pnv(const char* name, T v)
{
    printf("%s: ",name);
    std::cout<<v<<std::endl;
}

template <class T>
T t_min(T a, T b)
{
    if(a<b) return a;
    return b;
}

template <class T>
T sq(T a)
{
    return a*a;
}


inline void mfp_to_flt(flt* dst, mfp* src, s32 n_items)
{
  for(s32 i = 0; i < n_items; i++)
    *dst++ = *src++;
}

inline void mint_to_u8(u8* dst, mint* src, s32 n_items)
{
  for(s32 i = 0; i < n_items; i++)
    *dst++ = *src++;
}


void quat_to_rpy(Quaternionf q, Eigen::Matrix<fpt,3,1>& rpy)
{
  //from my MATLAB implementation

  //edge case!
  fpt as = t_min(2.*(q.w()*q.y() - q.x()*q.z()),.99999);
  rpy(0) = atan2(2.f*(q.x()*q.y()+q.w()*q.z()),sq(q.w()) + sq(q.x()) - sq(q.y()) - sq(q.z()));
  rpy(1) = asin(as);
  rpy(2) = atan2(2.f*(q.y()*q.z()+q.w()*q.x()),sq(q.w()) - sq(q.x()) - sq(q.y()) + sq(q.z()));
  // std::cout << "MPC solver rpy: " << rpy(0) << " " << rpy(1) << " " << rpy(2) << std::endl;
}

s8 near_zero(fpt a)
{
  return (a < 0.01 && a > -.01) ;
}

s8 near_one(fpt a)
{
  return near_zero(a-1);
}

inline Eigen::Matrix<fpt,3,3> cross_mat(Eigen::Matrix<fpt,3,3> I_inv, Eigen::Matrix<fpt,3,1> r)
{
  Eigen::Matrix<fpt,3,3> cm;
  cm << 0.f, -r(2), r(1),
    r(2), 0.f, -r(0),
    -r(1), r(0), 0.f;
  return I_inv * cm;
}


void matrix_to_real(qpOASES::real_t* dst, Eigen::Matrix<fpt,Dynamic,Dynamic> src, s16 rows, s16 cols)
{
  s32 a = 0;
  for(s16 r = 0; r < rows; r++)
  {
    for(s16 c = 0; c < cols; c++)
    {
      dst[a] = src(r,c);
      a++;
    }
  }
}



#endif

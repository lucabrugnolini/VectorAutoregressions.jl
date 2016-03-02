# Returns Magnus and Neudecker's duplication matrix of size n
function duplication(n::Int64)
  a = tril(ones(n,n))
  i = find(a)
  a[i] = 1:size(i,1)
  a = a + tril(a,-1)'
  j = trunc(Integer, vec(a))
  m = (n*(n+1)/2)
  m = trunc(Integer,m)
  d = zeros(n*n,m)
  for r = 1:size(d,1)
      d[r, j[r]] = 1
  end
  return d
end

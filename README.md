# Лабораторная работа №3
## Задание
<p align="justify">Разработать программу на языке CUDA С в соответствии с вариантами заданий к лабораторной работе, приведенным в таблице 1.</p>
<p align="justify">Написать программу на CUDA C, используя библиотеку CUBLAS.  Сравнить время вычислений с предыдущими лабораторными работами.</p>
<p align="justify">Документацию по библиотеке cublas взять из http://docs.nvidia.com/cuda/cublas/.</p>
<p align="justify">Также вычислить ускорение и эффективность. Эффективность определить с помощью профилировщика NVIDIA Visual Profiler.</p>
<table border="1">
  <caption>Вариант задания</caption>
  <tr>
    <th>№</th>
    <th>Размерность массива или матрицы</th>
    <th>Тип данных</th>
    <th>Описание задания</th>
  </tr>
  <tr align="center">
    <td>1</td>
    <td>10 x 10, 10 000 x 10 000</td>
    <td>float</td>
    <td>Вычислить матрицу C = ‖A‖ * A + B - A, ‖A‖ –  евклидова норма матрицы</td>
  </tr>
 </table>
 
 <table border="1">
  <caption>Время работы алгоритма</caption>
  <tr>
    <th>Размерность массива или матрицы</th>
    <th>Время линейного алгоритма, сек.</th>
    <th>Время параллельного алгоритма (только общая память), сек.</th>
    <th>Время параллельного алгоритма (с разделяемой памятью), сек.</th>
    <th>Время параллельного алгоритма (с библиотекой CUBLAS), сек.</th>
  </tr>
  <tr align="center">
    <td>1 000 x 1 000</td>
    <td>0.08</td>
    <td>0.03</td>
    <td>0.0225</td>
    <td>0.12</td>
  </tr>
  <tr align="center">
    <td>5 000 x 5 000</td>
    <td>1.67</td>
    <td>0.4</td>
    <td>0.2163</td>
    <td>0.2</td>
  </tr>
  <tr align="center">
    <td>10 000 x 10 000</td>
    <td>6.86</td>
    <td>1.55</td>
    <td>0.8041</td>
    <td>0.45</td>
  </tr>
  <tr align="center">
    <td>15 000 x 15 000</td>
    <td>12.9</td>
    <td>3.78</td>
    <td>2.07</td>
    <td>0.89</td>
  </tr>
 </table> 
 
<table border="1">
  <caption>Ускорение</caption>
  <tr>
    <th>Размерность массива или матрицы</th>
    <th>Ускорение (только общая память)</th>
    <th>Ускорение (с разделяемой памятью)</th>
    <th>Ускорение (с библиотекой CUBLAS)</th>
  </tr>
  <tr align="center">
    <td>1 000 x 1 000</td>
    <td>2.67</td>
    <td>3.56</td>
    <td>0.6</td>
  </tr>
  <tr align="center">
    <td>5 000 x 5 000</td>
    <td>4.17</td>
    <td>7.72</td>
    <td>8.35</td>
  </tr>
  <tr align="center">
    <td>10 000 x 10 000</td>
    <td>4.42</td>
    <td>8.53</td>
    <td>15.24</td>
  </tr>
  <tr align="center">
    <td>15 000 x 15 000</td>
    <td>3.42</td>
    <td>6.23</td>
    <td>14.49</td>
  </tr>
 </table> 

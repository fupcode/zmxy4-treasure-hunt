use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// BinaryHeap 是最大堆，反转全部字段以保持 Python 元组的最小堆顺序。
#[derive(Clone, Copy)]
struct Entry {
    priority: f64,
    remaining: i32,
    cost: f64,
    x: usize,
    y: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .total_cmp(&self.priority)
            .then_with(|| other.remaining.cmp(&self.remaining))
            .then_with(|| other.cost.total_cmp(&self.cost))
            .then_with(|| other.x.cmp(&self.x))
            .then_with(|| other.y.cmp(&self.y))
    }
}

fn path_to(mut point: usize, parents: &[usize], cols: usize) -> Vec<(usize, usize)> {
    let mut path = Vec::new();
    while parents[point] != usize::MAX {
        path.push((point % cols, point / cols));
        point = parents[point];
    }
    path.reverse();
    path
}

#[allow(clippy::too_many_arguments)]
fn search(
    rows: usize,
    cols: usize,
    maze: &[bool],
    start: (usize, usize),
    targets: &[f64],
    goal_distance: &[i32],
    distance: &[f64],
    hate_factor: f64,
    min_step_cost: f64,
    danger: Option<&[f64]>,
    preferred_end: Option<(isize, isize)>,
    stride: usize,
    allow_partial: bool,
    cost_map: Option<&[f64]>,
) -> Vec<(usize, usize)> {
    let start_index = start.1 * cols + start.0;
    let mut costs = vec![f64::INFINITY; maze.len()];
    let mut parents = vec![usize::MAX; maze.len()];
    costs[start_index] = 0.0;
    let mut queue = BinaryHeap::new();
    queue.push(Entry {
        priority: f64::from(goal_distance[start_index]) * min_step_cost,
        remaining: goal_distance[start_index],
        cost: 0.0,
        x: start.0,
        y: start.1,
    });
    let mut best = None;
    let mut best_value = f64::NEG_INFINITY;
    let mut safer = start_index;
    let directions = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    while let Some(current) = queue.pop() {
        let index = current.y * cols + current.x;
        if current.cost != costs[index] {
            continue;
        }
        if let Some(risk) = danger {
            if risk[index] < risk[safer] {
                safer = index;
            }
        }
        let value = targets[index];
        if value > 0.0 {
            if danger.is_some() {
                return path_to(index, &parents, cols);
            }
            if value > best_value {
                best = Some(index);
                best_value = value;
            }
            if value > 0.8 {
                return path_to(best.unwrap(), &parents, cols);
            }
        }
        for (dx, dy) in directions {
            let nx = current.x as isize + dx * stride as isize;
            let ny = current.y as isize + dy * stride as isize;
            if nx < 0 || ny < 0 || nx >= cols as isize || ny >= rows as isize {
                continue;
            }
            let neighbor = ny as usize * cols + nx as usize;
            if maze[neighbor] {
                continue;
            }
            let mut step_cost = 0.0;
            let mut blocked = false;
            for offset in 1..=stride {
                let x = (current.x as isize + dx * offset as isize) as usize;
                let y = (current.y as isize + dy * offset as isize) as usize;
                let cell = y * cols + x;
                if maze[cell] {
                    blocked = true;
                    break;
                }
                step_cost += 1.0 + hate_factor / (distance[cell] + 1.0);
                if let Some(risk) = danger {
                    step_cost += risk[cell];
                }
                if let Some(extra) = cost_map {
                    step_cost += extra[cell];
                }
            }
            if blocked {
                continue;
            }
            let new_cost = current.cost + step_cost;
            if new_cost < costs[neighbor] {
                costs[neighbor] = new_cost;
                parents[neighbor] = index;
                let preference = preferred_end.map_or(0.0, |(x, y)| {
                    1e-6 * ((nx - x).abs() + (ny - y).abs()) as f64
                });
                let remaining = goal_distance[neighbor];
                let priority = new_cost + f64::from(remaining) * min_step_cost + preference;
                // Python 中 priority 为 NumPy float64，其 round 使用缩放后 ties-to-even。
                queue.push(Entry {
                    priority: (priority * 1e9).round_ties_even() / 1e9,
                    remaining,
                    cost: new_cost,
                    x: nx as usize,
                    y: ny as usize,
                });
            }
        }
    }
    if let Some(point) = best {
        path_to(point, &parents, cols)
    } else if danger.is_some() && allow_partial {
        path_to(safer, &parents, cols)
    } else {
        Vec::new()
    }
}

#[pyfunction]
#[pyo3(signature = (maze, start, targets, goal_distance, distance, hate_factor, min_step_cost,
                    danger, preferred_end, stride, allow_partial, cost_map))]
#[allow(clippy::too_many_arguments)]
fn region_path(
    _py: Python<'_>,
    maze: PyReadonlyArray2<'_, bool>,
    start: (usize, usize),
    targets: PyReadonlyArray2<'_, f64>,
    goal_distance: PyReadonlyArray2<'_, i32>,
    distance: PyReadonlyArray2<'_, f64>,
    hate_factor: f64,
    min_step_cost: f64,
    danger: Option<PyReadonlyArray2<'_, f64>>,
    preferred_end: Option<(isize, isize)>,
    stride: usize,
    allow_partial: bool,
    cost_map: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Vec<(usize, usize)>> {
    let view = maze.as_array();
    let (rows, cols) = (view.nrows(), view.ncols());
    if stride == 0 || stride > isize::MAX as usize || start.0 >= cols || start.1 >= rows {
        return Err(PyValueError::new_err(
            "start must be inside maze and stride must be positive",
        ));
    }
    let shape = [rows, cols];
    if targets.as_array().shape() != shape
        || goal_distance.as_array().shape() != shape
        || distance.as_array().shape() != shape
        || danger
            .as_ref()
            .is_some_and(|a| a.as_array().shape() != shape)
        || cost_map
            .as_ref()
            .is_some_and(|a| a.as_array().shape() != shape)
    {
        return Err(PyValueError::new_err(
            "all maps must have the same shape as maze",
        ));
    }
    // 持有 GIL 和只读借用守卫，直接读取 NumPy 内存，不复制输入数组。
    let danger_slice = danger.as_ref().map(|a| a.as_slice()).transpose()?;
    let cost_slice = cost_map.as_ref().map(|a| a.as_slice()).transpose()?;
    Ok(search(
        rows,
        cols,
        maze.as_slice()?,
        start,
        targets.as_slice()?,
        goal_distance.as_slice()?,
        distance.as_slice()?,
        hate_factor,
        min_step_cost,
        danger_slice,
        preferred_end,
        stride,
        allow_partial,
        cost_slice,
    ))
}

#[pymodule]
fn _maze_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(region_path, module)?)?;
    Ok(())
}

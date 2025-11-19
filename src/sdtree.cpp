#include "rdr/sdtree.h"

#include "rdr/film.h"
#include "rdr/render.h"

RDR_NAMESPACE_BEGIN

namespace detail_ {
Float GammaCorrection(Float value) {
  if (value <= 0.04045F) {
    return value / 12.92F;
  } else {
    return Pow(value + 0.055F / 1.055F, 2.4F);
  }
}

Float XyzToLabHelper(Float value) {
  if (value > 0.008856F) {
    return powf(value, 1.0F / 3.0F);
  } else {
    return (7.787F * value) + (16.0F / 116.0F);
  }
}

Float RgbToCieLabLightness(const Vec3f &rgb) {
  // Apply gamma correction
  const Float r = GammaCorrection(rgb.x);
  const Float g = GammaCorrection(rgb.y);
  const Float b = GammaCorrection(rgb.z);

  // Convert RGB to XYZ
  const Float x = r * 0.4124564F + g * 0.3575761F + b * 0.1804375F;
  const Float y = r * 0.2126729F + g * 0.7151522F + b * 0.0721750F;
  const Float z = r * 0.0193339F + g * 0.1191920F + b * 0.9503041F;

  // Normalize XYZ values
  const Float xn = x / 0.95047F;
  const Float yn = y;
  const Float zn = z / 1.08883F;

  // Convert XYZ to Lab
  return 116.0F * XyzToLabHelper(yn) - 16.0F;
}
}  // namespace detail_

//===----------------------------------------------------------------------===//
// DirectionalQuadTree Impl
//===----------------------------------------------------------------------===//

void DirectionalQuadTree::commitLocalPoint(
    IndexType node_index, const Vec2f &point, const Vec3f &weight) {
  auto &leaf_node = nodes[findLeaf(node_index, point)];

  // Power filter
  const Vec3f p_weight = Pow(weight, filter_power);
  const Float inc =
      detail_::RgbToCieLabLightness(p_weight) * leaf_node.bound.getVolume();

  while (true) {
    QuadNode &current_node = nodes[node_index];
    current_node.weight += inc;
    if (current_node.is_leaf) {
      current_node.flux += leaf_node.bound.getVolume() * weight;
      return;
    }

    node_index = current_node.children[child(point, current_node.bound)];
  }
}

Vec2f DirectionalQuadTree::sampleLocalPoint(const IndexType &node_index,
    Sampler &sampler, Float &pdf, IndexType &discrete_index, int depth) const {
  const QuadNode &current_node = nodes[node_index];

  if (current_node.is_leaf) {
    // Sample a point in the leaf
    const Vec2f &low_bnd   = current_node.bound.low_bnd;
    const Vec2f &upper_bnd = current_node.bound.upper_bnd;
    const Vec2f &center    = (low_bnd + upper_bnd) / static_cast<Float>(2.0);
    const Vec2f &ustep     = Vec2f(center.x - low_bnd.x, 0);
    const Vec2f &vstep     = Vec2f(0, upper_bnd.y - center.y);

    const Float u  = sampler.get1D();
    const Float v  = sampler.get1D();
    pdf            = 1.0_F / current_node.bound.getVolume();
    discrete_index = node_index;
    return low_bnd + u * ustep + v * vstep;
  }

  // Populate the distribution
  std::array<Float, 4> weights{};
  for (int i = 0; i < 4; ++i)
    weights[i] = nodes[current_node.children[i]].weight;
  Distribution1D dist(weights.data(), 4);

  Float local_pmf = NAN;
  Float local_pdf = NAN;
  const int sampled_leaf_index =
      dist.sampleDiscrete(sampler.get1D(), &local_pmf);
  assert(0 <= sampled_leaf_index && sampled_leaf_index < 4);

  const Vec2f sampled_point =
      sampleLocalPoint(current_node.children[sampled_leaf_index], sampler,
          local_pdf, discrete_index, depth + 1);
  pdf = local_pmf * local_pdf;
  return sampled_point;
}

float DirectionalQuadTree::pdfLocalPoint(const Vec2f &point) const {
  const IndexType leaf_index = findLeaf(root_index, point);
  const QuadNode &leaf_node  = nodes[leaf_index];
  return leaf_node.weight / nodes[root_index].weight /
         leaf_node.bound.getVolume();
}

void DirectionalQuadTree::splitOrPruneNode(
    const IndexType &node_index, Float split_ratio, int depth) {
  const Float total_weight = nodes[root_index].weight;
  const Float node_weight  = nodes[node_index].weight;

  // If the node is too small and contains child, prune it
  if (node_weight < split_ratio * total_weight) {
    if (!nodes[node_index].is_leaf) {
      // Prune the children
      for (IndexType &child_index : nodes[node_index].children)
        child_index = INVALID_INDEX;
      nodes[node_index].is_leaf = true;
    }

    return;
  }

  if (nodes[node_index].is_leaf) {
    // Should split
    nodes[node_index].is_leaf = false;

    // do not use reference. Since nodes may be reallocated
    const Vec2f low_bnd   = nodes[node_index].bound.low_bnd;
    const Vec2f upper_bnd = nodes[node_index].bound.upper_bnd;
    const Vec2f center    = (low_bnd + upper_bnd) / static_cast<Float>(2.0);
    const Vec2f ustep     = Vec2f(center.x - low_bnd.x, 0);
    const Vec2f vstep     = Vec2f(0, upper_bnd.y - center.y);

    /// ---------, -----------
    /// | 2 | 3 |  | 10 | 11 |
    /// | 0 | 1 |  | 00 | 10 |
    /// ---------  -----------
    /// which can be determined by bit operation

    const Float child_weight = node_weight / 4.0_F;
    const Vec3f node_flux    = nodes[node_index].flux / 4.0_F;
    // Left bottom corner
    nodes.push_back(
        QuadNode{true, child_weight, TAABB<Vec2f>(low_bnd, center), node_flux});
    // Right bottom coner
    nodes.push_back(QuadNode{true, child_weight,
        TAABB<Vec2f>(low_bnd + ustep, center + ustep), node_flux});
    // Left top corner
    nodes.push_back(QuadNode{true, child_weight,
        TAABB<Vec2f>(low_bnd + vstep, center + vstep), node_flux});
    // Right top corner
    nodes.push_back(QuadNode{
        true, child_weight, TAABB<Vec2f>(center, upper_bnd), node_flux});

    nodes[node_index].children[0] = nodes.size() - 4;
    nodes[node_index].children[1] = nodes.size() - 3;
    nodes[node_index].children[2] = nodes.size() - 2;
    nodes[node_index].children[3] = nodes.size() - 1;

    // Recursively split children
    for (const IndexType &child_index : nodes[node_index].children)
      splitOrPruneNode(child_index, split_ratio, depth + 1);
  } else {
    for (const IndexType &child_index : nodes[node_index].children)
      splitOrPruneNode(child_index, split_ratio, depth + 1);
  }
}

DirectionalQuadTree::IndexType DirectionalQuadTree::findLeaf(
    IndexType node_index, const Vec2f &point) const {
  AssertAllNonNegative(node_index);

  while (true) {
    const QuadNode &current_node = nodes[node_index];
    if (current_node.is_leaf) return node_index;

    // Ask childrens
    node_index = current_node.children[child(point, current_node.bound)];
  }
}

void DirectionalQuadTree::visualizeToImage(const fs::path &path) const {
  // Prepare the canvas
  auto canvas = NativeRender::prepareDebugCanvas({1024, 1024});

  // Render the quadtree
  recursiveVisualizer(root_index, canvas, 0);

  // Render the quadtree
  canvas.exportImageToFile(FileResolver::resolveToAbs(path));
}

/// Paint the quad tree to an image
void DirectionalQuadTree::recursiveVisualizer(
    const IndexType &node_index, Film &canvas, int depth) const {
  const QuadNode &current_node = nodes[node_index];
  const auto resolution        = canvas.getResolution();
  const auto &bound            = current_node.bound;
  const Vec2i &low_bnd         = Cast<int>(bound.low_bnd * resolution);
  const Vec2i &upper_bnd       = Cast<int>(bound.upper_bnd * resolution);

  if (current_node.is_leaf) {
    // Do the painting
    for (int x = low_bnd.x; x < upper_bnd.x; ++x)
      for (int y = low_bnd.y; y < upper_bnd.y; ++y)
        canvas.commitSample({x + 0.5_F, y + 0.5_F},
            Vec3f{pdfLocalPoint(Vec2f(x + 0.5_F, y + 0.5_F) / resolution)});
    // canvas.commitSample({x + 0.5_F, y + 0.5_F},
    //     Vec3f{current_node.flux} / current_node.bound.getVolume() /
    //         nodes[root_index].weight);
    return;
  }

  // Enable to visualize the split lines
  // Paint the split line
  /*
  const Vec3f line_color{100};
  const Vec2f split_val = (low_bnd + upper_bnd) / 2.0_F;
  for (int x = low_bnd.x; x < upper_bnd.x; ++x)
    canvas.commitSample({x, split_val.y}, line_color);
  for (int y = low_bnd.y; y < upper_bnd.y; ++y)
    canvas.commitSample({split_val.x, y}, line_color);
  */

  // Ask childrens
  for (const IndexType &child_index : current_node.children)
    recursiveVisualizer(child_index, canvas, depth + 1);
}

void DirectionalQuadTree::sanityCheck(
    const IndexType &node_index, int depth) const {
  // Check if the node is valid
  const QuadNode &current_node = nodes[node_index];

  if (!current_node.is_leaf) {
    Float total_weight = 0.0_F;
    for (const IndexType &child_index : current_node.children) {
      assert(child_index != INVALID_INDEX);
      assert(child_index < nodes.size());

      total_weight += nodes[child_index].weight;
      sanityCheck(child_index, depth + 1);
    }

    assert(abs(current_node.weight - total_weight) < EPS);
    return;
  } else {
    assert(current_node.weight >= 0.0_F);
    assert(current_node.weight < SPLIT_RATIO * nodes[root_index].weight);
    return;
  }
}

//===----------------------------------------------------------------------===//
// SpatialBinaryTree Impl
//===----------------------------------------------------------------------===//

SpatialBinaryTree::SpatialBinaryTree(
    AABB bound, std::pmr::memory_resource *upstream)
    : root_bound(boundTransformation(bound)),
      upstream(upstream),
      leaf_nodes(upstream),
      internal_nodes(upstream),
      internal_nodes_update_mutex(make_ref<std::mutex>()),
      leaf_nodes_update_mutex(make_ref<std::mutex>()) {
  initNodes();
}

SpatialBinaryTree::SpatialBinaryTree(const SpatialBinaryTree &other)
    : root_bound(other.root_bound),
      upstream(other.upstream),
      leaf_nodes(other.leaf_nodes, upstream),
      internal_nodes(other.internal_nodes, upstream),
      internal_nodes_update_mutex(make_ref<std::mutex>()),
      leaf_nodes_update_mutex(make_ref<std::mutex>()) {}

void SpatialBinaryTree::clear() {
  leaf_nodes.clear();
  internal_nodes.clear();
  initNodes();
}

void SpatialBinaryTree::optimize(uint64_t split_threshold) {
  std::deque<uint64_t> leaf_node_indices;
  for (int i = 0; i < leaf_nodes.size(); ++i) leaf_node_indices.push_back(i);

  // BFS all leaf nodes
  while (!leaf_node_indices.empty()) {
    const uint64_t leaf_node_index = leaf_node_indices.front();
    leaf_node_indices.pop_front();

    // Check if the leaf node is valid
    if (leaf_nodes[leaf_node_index].num_samples < split_threshold) {
      leaf_nodes[leaf_node_index].flushSamples();
      continue;
    }

    // Split the leaf node
    splitAndPushdownNode(leaf_node_index);

    // Push the children nodes
    const auto &internal_node = internal_nodes.back();
    assert(internal_node.is_child_leaf[0] && internal_node.is_child_leaf[1]);
    leaf_node_indices.push_back(internal_node.children[0]);
    leaf_node_indices.push_back(internal_node.children[1]);
  }

  assert(std::all_of(leaf_nodes.begin(), leaf_nodes.end(),
      [](const LeafNode &node) -> bool { return node.num_samples == 0; }));
}

void SpatialBinaryTree::commitSample(const Vec3f &position,
    const Vec3f &direction, const Vec3f &measurement, Sampler &sampler,
    int thread_id) {
  (void)(thread_id);

  // find and disturb the samples to the leaf node
  const auto leaf_index = findClosestLeafIndex(position);
  assert(leaf_index < leaf_nodes.size());
  const AABB leaf_bound   = leaf_nodes[leaf_index].bound;
  const Vec3f leaf_extent = leaf_bound.getExtent();
  const Vec3f random_vector =
      1 - 2 * Vec3f{sampler.get1D(), sampler.get1D(), sampler.get1D()};
  const Vec3f disturbed_position = position + leaf_extent * random_vector;

  // Finally commit the disturbed sample
  leaf_nodes[findClosestLeafIndex(disturbed_position)].commitSample(
      Sample{position, direction, measurement});
}

const SpatialBinaryTree::LeafNode &SpatialBinaryTree::findClosestLeaf(
    const Vec3f &position) const {
  const IndexType &leaf_index = findClosestLeafIndex(position);
  assert(leaf_index < leaf_nodes.size());
  return leaf_nodes[leaf_index];
}

void SpatialBinaryTree::initNodes() {
  assert(leaf_nodes.empty() && internal_nodes.empty());

  // Create the initial condition, one internal node points to two leaf nodes
  const int split_dim = 0;
  internal_nodes.push_back(InternalNode(AABB{}));
  InternalNode &root_node    = internal_nodes.back();
  root_node.is_child_leaf[0] = true;
  root_node.is_child_leaf[1] = true;
  root_node.children[0]      = 0;
  root_node.children[1]      = 1;
  root_node.bound            = root_bound;
  root_node.split_dim        = split_dim;
  root_node.split_val =
      (root_bound.low_bnd[split_dim] + root_bound.upper_bnd[split_dim]) / 2.0_F;

  leaf_nodes.push_back(LeafNode(upstream));
  leaf_nodes.push_back(LeafNode(upstream));

  leaf_nodes[0].parent                     = 0;
  leaf_nodes[0].depth                      = 1;
  leaf_nodes[0].bound                      = root_bound;
  leaf_nodes[0].bound.upper_bnd[split_dim] = root_node.split_val;

  leaf_nodes[1].parent                   = 0;
  leaf_nodes[1].depth                    = 1;
  leaf_nodes[1].bound                    = root_bound;
  leaf_nodes[1].bound.low_bnd[split_dim] = root_node.split_val;

  leaf_nodes[0].num_samples = 256;
  leaf_nodes[1].num_samples = 256;

  // Bootstrap the tree
  optimize(1);
}

void SpatialBinaryTree::splitAndPushdownNode(const IndexType &leaf_node_index) {
  // Flush the samples
  leaf_nodes[leaf_node_index].quad_tree.syncSamples();

  // Create the replacing internal node and the new leaf node
  internal_nodes.emplace_back(leaf_nodes[leaf_node_index].bound);
  leaf_nodes.emplace_back(leaf_nodes[leaf_node_index]);

  LeafNode &current_leaf_node = leaf_nodes[leaf_node_index];
  const int dim               = current_leaf_node.depth % 3;

  InternalNode &parent_node = internal_nodes[current_leaf_node.parent];
  assert(parent_node.children[0] == leaf_node_index ||
         parent_node.children[1] == leaf_node_index);
  const bool is_left_child = (parent_node.is_child_leaf[0] &&
                              parent_node.children[0] == leaf_node_index);

  // Update the parent_node's state
  parent_node.is_child_leaf[is_left_child ? 0 : 1] = false;
  parent_node.children[is_left_child ? 0 : 1]      = internal_nodes.size() - 1;

  // Update the new internal node
  InternalNode &internal_node = internal_nodes.back();
  internal_node.bound         = current_leaf_node.bound;
  internal_node.split_dim     = dim;
  internal_node.split_val     = (current_leaf_node.bound.low_bnd[dim] +
                                current_leaf_node.bound.upper_bnd[dim]) /
                            2.0_F;  // the median position
  internal_node.is_child_leaf[0] = true;
  internal_node.is_child_leaf[1] = true;
  internal_node.children[0]      = leaf_node_index;
  internal_node.children[1]      = leaf_nodes.size() - 1;

  // Update the samples in the two leaf nodes
  const Float split_val       = internal_node.split_val;
  LeafNode &first_child_node  = leaf_nodes[leaf_node_index];
  LeafNode &second_child_node = leaf_nodes.back();

  first_child_node.depth += 1;
  first_child_node.parent               = internal_nodes.size() - 1;
  first_child_node.bound.upper_bnd[dim] = split_val;
  first_child_node.num_samples /= 2;

  second_child_node.depth += 1;
  second_child_node.parent             = internal_nodes.size() - 1;
  second_child_node.bound.low_bnd[dim] = split_val;
  second_child_node.num_samples /= 2;
}

SpatialBinaryTree::IndexType SpatialBinaryTree::findClosestLeafIndex(
    const Vec3f &position) const {
  bool is_leaf         = false;
  IndexType node_index = 0;  // in internal nodes

  while (true) {
    if (is_leaf)
      // Commit the sample to the leaf node
      return node_index;
    else {
      // Check if the internal node is a leaf node
      const InternalNode &internal_node = internal_nodes[node_index];
      const bool is_left_child =
          (position[internal_node.split_dim] < internal_node.split_val);
      is_leaf    = internal_node.is_child_leaf[is_left_child ? 0 : 1];
      node_index = internal_node.children[is_left_child ? 0 : 1];
    }
  }
}

RDR_NAMESPACE_END

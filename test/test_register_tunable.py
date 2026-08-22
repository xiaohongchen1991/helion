from __future__ import annotations

import re
import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfSharedMemoryLessThan
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
from helion.autotuner import IntegerFragment
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl
from helion.language import loops


@onlyBackends(["triton", "cute"])
class TestRegisterTunable(RefEagerTestBase, TestCase):
    maxDiff = 10000

    def test_power_of_two_fragment_basic(self):
        @helion.kernel(autotune_effort="none")
        def kernel_with_tunable(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)

            # Register a tunable parameter for block size
            block_size = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))

            for tile_n in hl.tile([n], block_size=[block_size * 2]):
                out[tile_n] = x[tile_n] * 2.0

            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(kernel_with_tunable, (x,))
        expected = x * 2.0
        torch.testing.assert_close(result, expected)
        self.assertIsInstance(
            self.getUserDefinedTunable(
                kernel_with_tunable.bind((x,)).config_spec.user_defined_tunables, "foo"
            ),
            PowerOfTwoFragment,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @patch.object(loops, "_supports_warp_specialize", lambda: False)
    def test_integer_fragment(self):
        @helion.kernel()
        def kernel_with_int_param(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            # Register an integer tunable parameter
            multiplier = hl.register_tunable("multiplier", IntegerFragment(1, 10, 3))
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * multiplier
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            kernel_with_int_param, (x,), block_size=64, multiplier=4
        )
        expected = x * 4
        torch.testing.assert_close(result, expected)
        default_config = repr(
            kernel_with_int_param.bind((x,)).config_spec.default_config()
        )
        self.assertIn("multiplier=3", default_config)

    def test_enum_fragment(self):
        @helion.kernel(config={"operation": 2})
        def kernel_with_enum(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)

            # Register an enum tunable parameter
            operation = hl.register_tunable("operation", EnumFragment((1, 2, 4)))

            for tile_n in hl.tile([n], block_size=[64]):
                out[tile_n] = x[tile_n] * operation

            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        result = kernel_with_enum(x)
        expected = x * 2.0
        torch.testing.assert_close(result, expected)

    def test_tensor_allocated_with_block_size(self):
        @helion.kernel()
        def fn(x: torch.Tensor):
            m = x.size(0)
            block_m = hl.register_block_size(m)
            tiles_m = (m + block_m - 1) // block_m  # cdiv
            partial = torch.zeros(tiles_m, dtype=x.dtype, device=x.device)
            for tile in hl.tile(m, block_size=block_m):
                partial[tile.begin // block_m] = x[tile].sum()
            return partial.sum()

        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(fn, (x,), block_size=64)
        torch.testing.assert_close(result, x.sum())

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfSharedMemoryLessThan(
        86016, reason="num_stages=8 requires 86016 bytes of shared memory"
    )
    def test_matmul_split_k(self):
        """Test matmul_split_k kernel with register_tunable"""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[32, 64, 64],
                loop_orders=[[1, 2, 0]],
                num_warps=16,
                num_stages=8,
                indexing="block_ptr",
                split_k=64,
            )
        )
        def matmul_split_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.zeros(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
            k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
            for tile_m, tile_n, outer_k in hl.tile(
                [m, n, k], block_size=[None, None, k_block]
            ):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for inner_k in hl.tile(outer_k.begin, outer_k.end):
                    acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
                hl.atomic_add(out, [tile_m, tile_n], acc)
            return out

        m, k, n = 64, 4096, 64
        x = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)

        code, result = code_and_output(matmul_split_k, (x, y))
        expected = (
            (x.cpu().float() @ y.cpu().float()).to(result.dtype).to(result.device)
        )
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1)
        self.assertIsInstance(
            self.getUserDefinedTunable(
                matmul_split_k.bind((x, y)).config_spec.user_defined_tunables, "split_k"
            ),
            PowerOfTwoFragment,
        )

    @skipIfRefEager("compile_config not supported in ref eager mode")
    def test_tunable_passed_as_constexpr(self):
        """A tunable used on device is passed as constexpr, not a runtime scalar.

        The value is constant for the config being compiled, so marking it
        constexpr lets Triton drop the dead side of a branch that selects on it.
        As a runtime scalar both sides get compiled instead: for a swap-AB GEMM
        that keeps a second matmul alive, costing registers and occupancy for
        code that never runs.
        """

        @helion.kernel(static_shapes=True)
        def matmul_swap_ab(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            M, K = a.size()
            _, N = b.size()
            out = torch.empty([M, N], dtype=a.dtype, device=a.device)
            block_k = hl.register_block_size(K)
            swap_ab = hl.register_tunable("swap_ab", BooleanFragment())
            for tile_m, tile_n in hl.tile([M, N]):
                if swap_ab:
                    acc_swap = hl.zeros([tile_n, tile_m], dtype=torch.float32)
                    for tile_k in hl.tile(K, block_size=block_k):
                        a_blk = hl.load(
                            a, [tile_m.index[None, :], tile_k.index[:, None]]
                        )
                        b_blk = hl.load(
                            b, [tile_k.index[None, :], tile_n.index[:, None]]
                        )
                        acc_swap = hl.dot(
                            b_blk, a_blk, acc=acc_swap, out_dtype=torch.float32
                        )
                    out[tile_m, tile_n] = acc_swap.t().to(out.dtype)
                else:
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(K, block_size=block_k):
                        acc = hl.dot(
                            a[tile_m, tile_k],
                            b[tile_k, tile_n],
                            acc=acc,
                            out_dtype=torch.float32,
                        )
                    out[tile_m, tile_n] = acc.to(out.dtype)
            return out

        M, K, N = 32, 256, 128
        a = torch.randn(M, K, device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn(K, N, device=DEVICE, dtype=HALF_DTYPE)
        expected = a @ b

        bound = matmul_swap_ab.bind((a, b))
        for swap_ab in [True, False]:
            config = helion.Config(block_sizes=[32, 64, 64], swap_ab=swap_ab)
            code = bound.to_triton_code(config)
            params = re.search(r"^def _helion_matmul_swap_ab\((.*)\):$", code, re.M)
            assert params is not None
            self.assertIn("swap_ab: tl.constexpr", params.group(1))
            self.assertIn("if swap_ab:", code)
            result = bound.compile_config(config)(a, b)
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
